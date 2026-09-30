"""Stage 1: pretrain a base model to predict the next character of a text corpus."""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from .checkpoint import load_checkpoint, mark_finished, save_checkpoint
from .config import to_dict
from .data import encode_text_bytes, expand_paths, get_text_batch, read_text_bytes, text_chars
from .model import CharTransformerLanguageModel, ModelConfig
from .tokenizer import DEFAULT_CHARS, CharTokenizer
from .training import (MetricsLogger, TrainConfig, clear_state, load_state, make_optimizer, metrics_path,
                       optimizer_step, resolve_device, save_state, set_seed, setup_precision, skip_if_finished)


@dataclass
class PretrainConfig(TrainConfig):
    # the text to train on: a file, a directory of .txt files, a glob pattern like "data/*.txt", or a list of these
    # (python -m charlm prepare-data downloads larger datasets)
    data_path: Union[str, List[str]] = "data/input.txt"
    out_path: str = "checkpoints/base.pt"
    init_from: Optional[str] = None  # continue training this checkpoint instead of starting from scratch
    model: ModelConfig = field(default_factory=ModelConfig)  # ignored when init_from is set
    val_fraction: float = 0.1  # the last part of each file is held out for validation...
    max_val_chars: int = 0  # ...up to this many characters per file (0 = no limit; a big corpus needs little)
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200  # batches averaged for each loss estimate
    patience: int = 3  # stop after this many evaluations without val loss improvement (0 = never stop early)
    sample_tokens: int = 300  # length of the text sample printed at the end


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    """ average loss over eval_iters random batches of each split """
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_text_batch(data, cfg.batch_size, model.config.block_size, device)
            losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def load_splits(paths, tokenizer, val_fraction, max_val_chars):
    """ {'train', 'val'} token tensors: the end of each file is its validation part """
    train, val = [], []
    for path in paths:
        data = encode_text_bytes(tokenizer, read_text_bytes(path))
        n_val = len(data) - int((1 - val_fraction) * len(data))
        if max_val_chars:
            n_val = min(n_val, max_val_chars)
        train.append(data[:len(data) - n_val])
        val.append(data[len(data) - n_val:])
    join = lambda parts: parts[0] if len(parts) == 1 else torch.cat(parts)
    return {'train': join(train), 'val': join(val)}


def pretrain(cfg):
    """ trains on cfg.data_path, keeps the checkpoint with the lowest val loss, returns (model, tokenizer) """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    done = skip_if_finished(cfg, device)
    if done:
        return done
    paths = expand_paths(cfg.data_path)

    if cfg.init_from:
        model, tokenizer, _ = load_checkpoint(cfg.init_from, device)
    else:
        chars = set(DEFAULT_CHARS)
        for path in paths:
            chars |= text_chars(read_text_bytes(path))
        tokenizer = CharTokenizer(chars)
        cfg.model.vocab_size = tokenizer.vocab_size
        model = CharTransformerLanguageModel(cfg.model).to(device)
    print(f"device {device} | {model.num_params() / 1e6:.2f}M parameters | vocab {tokenizer.vocab_size}")

    splits = load_splits(paths, tokenizer, cfg.val_fraction, cfg.max_val_chars)
    for name, data in splits.items():
        if len(data) <= model.config.block_size:
            raise ValueError(f"the {name} split has {len(data)} characters, too few for block_size "
                             f"{model.config.block_size}")
    print(f"{len(paths)} file(s) | {len(splits['train']):,} train / {len(splits['val']):,} val characters")
    fingerprint = [[path, os.path.getsize(path)] for path in paths]  # a resumed run must train on the same data

    optimizer = make_optimizer(model, cfg)
    scaler = setup_precision(cfg, device, model)
    best_val_loss, evals_without_improvement = float('inf'), 0
    start_iter, state = (load_state(cfg.out_path, model, optimizer, device, cfg, scaler) if cfg.resume
                         else (0, None))
    if state is not None:
        if state.get('data', fingerprint) != fingerprint:
            raise ValueError(f"the data files changed since the interrupted run was saved "
                             f"({state['data']} -> {fingerprint}); restore them or delete the .state.pt file")
        best_val_loss, evals_without_improvement = state['best_val_loss'], state['evals_without_improvement']
    logger = MetricsLogger(metrics_path(cfg.out_path), append=state is not None, keep_until=start_iter)

    for it in range(start_iter, cfg.max_iters + 1):  # the last iteration only evaluates the final update
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters) and not (state is not None and it == start_iter):
            losses = estimate_loss(model, splits, cfg, device)
            logger.log(it, train_loss=losses['train'], val_loss=losses['val'])
            # keep the model with the lowest val loss, and stop once it stops improving
            if losses['val'] < best_val_loss:
                best_val_loss, evals_without_improvement = losses['val'], 0
                save_checkpoint(cfg.out_path, model, tokenizer,
                                {'stage': 'pretrain', 'iter': it, 'val_loss': best_val_loss, 'config': to_dict(cfg)})
            else:
                evals_without_improvement += 1
                if cfg.patience and evals_without_improvement >= cfg.patience:
                    print(f"early stopping: val loss has not improved for {cfg.patience} evaluations")
                    break
            if (it // cfg.eval_interval) % cfg.state_every == 0:
                save_state(cfg.out_path, model, optimizer, it, cfg, scaler, best_val_loss=best_val_loss,
                           evals_without_improvement=evals_without_improvement, data=fingerprint)
        if it == cfg.max_iters:
            break

        x, y = get_text_batch(splits['train'], cfg.batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        optimizer_step(model, optimizer, loss, it, cfg, scaler)

    mark_finished(cfg.out_path)
    clear_state(cfg.out_path)
    model, tokenizer, _ = load_checkpoint(cfg.out_path, device)
    print(f"saved best model (val loss {best_val_loss:.4f}) to {cfg.out_path}")
    if cfg.sample_tokens:
        context = torch.tensor([tokenizer.encode('\n')], dtype=torch.long, device=device)
        print(tokenizer.decode(model.generate(context, cfg.sample_tokens)[0].tolist()))
    return model, tokenizer

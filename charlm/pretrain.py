"""Stage 1: pretrain a base model to predict the next character of a text corpus."""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import to_dict
from .data import get_text_batch, load_text
from .model import CharTransformerLanguageModel, ModelConfig
from .tokenizer import CharTokenizer
from .training import (MetricsLogger, TrainConfig, make_optimizer, metrics_path, optimizer_step, resolve_device,
                       set_seed)


@dataclass
class PretrainConfig(TrainConfig):
    data_path: str = "data/input.txt"  # plain text file to train on
    out_path: str = "checkpoints/base.pt"
    init_from: Optional[str] = None  # continue training this checkpoint instead of starting from scratch
    model: ModelConfig = field(default_factory=ModelConfig)  # ignored when init_from is set
    val_fraction: float = 0.1  # the last part of the text is held out for validation
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200  # batches averaged for each loss estimate
    patience: int = 3  # stop after this many evaluations without val loss improvement (0 = never stop early)
    sample_tokens: int = 300  # length of the text sample printed at the end
    resume: bool = False  # continue an interrupted run from its last evaluation (saved in *.state.pt)


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


def pretrain(cfg):
    """ trains on cfg.data_path, keeps the checkpoint with the lowest val loss, returns (model, tokenizer) """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    text = load_text(cfg.data_path)

    if cfg.init_from:
        model, tokenizer, _ = load_checkpoint(cfg.init_from, device)
    else:
        tokenizer = CharTokenizer.from_text(text)
        cfg.model.vocab_size = tokenizer.vocab_size
        model = CharTransformerLanguageModel(cfg.model).to(device)
    print(f"device {device} | {model.num_params() / 1e6:.2f}M parameters | vocab {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text, allow_special=False), dtype=torch.long)
    n = int((1 - cfg.val_fraction) * len(data))
    splits = {'train': data[:n], 'val': data[n:]}

    optimizer = make_optimizer(model, cfg)
    state_path = os.path.splitext(cfg.out_path)[0] + '.state.pt'
    start_iter, best_val_loss, evals_without_improvement = 0, float('inf'), 0
    if cfg.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        start_iter, best_val_loss, evals_without_improvement = (
            state['iter'], state['best_val_loss'], state['evals_without_improvement'])
        print(f"resuming from iteration {start_iter} (best val loss {best_val_loss:.4f})")
    logger = MetricsLogger(metrics_path(cfg.out_path), append=start_iter > 0)

    for it in range(start_iter, cfg.max_iters):
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters - 1) and not (it == start_iter > 0):
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
            # everything needed to continue from here if the run is interrupted
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                        'iter': it, 'best_val_loss': best_val_loss,
                        'evals_without_improvement': evals_without_improvement}, state_path)

        x, y = get_text_batch(splits['train'], cfg.batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        optimizer_step(model, optimizer, loss, it, cfg)

    if os.path.exists(state_path):
        os.remove(state_path)  # the run is finished, nothing left to resume
    model, tokenizer, _ = load_checkpoint(cfg.out_path, device)
    print(f"saved best model (val loss {best_val_loss:.4f}) to {cfg.out_path}")
    if cfg.sample_tokens:
        context = torch.tensor([tokenizer.encode('\n')], dtype=torch.long, device=device)
        print(tokenizer.decode(model.generate(context, cfg.sample_tokens)[0].tolist()))
    return model, tokenizer

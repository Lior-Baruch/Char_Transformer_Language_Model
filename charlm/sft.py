"""Stage 2: supervised fine-tuning (SFT) turns the base model into an instruction-following chat model.

The model is trained on (prompt, response) pairs in the chat template, with the loss computed only on the
response tokens: it learns to answer, not to predict the user's message.

With reasoning=true, the replies to math tasks first write a step-by-step scratchpad between <|think|> and
<|/think|> (see reasoning.py). The base model has no such tokens, so they are added to the tokenizer and the
model's vocabulary grows by two rows, which SFT then learns.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .chat import encode_chat_example, sample_replies
from .checkpoint import load_checkpoint, mark_finished, save_checkpoint
from .config import to_dict
from .data import load_text, pad_batch, read_jsonl
from .evaluation import evaluate_tasks
from .reasoning import format_response
from .tasks import ALL_TASKS, TaskSuite, eval_tasks
from .tokenizer import REASONING_TOKENS
from .training import (MetricsLogger, TrainConfig, clear_state, load_state, make_optimizer, metrics_path,
                       optimizer_step, resolve_device, save_state, set_seed, setup_precision, skip_if_finished)


@dataclass
class SFTConfig(TrainConfig):
    init_from: str = "checkpoints/base.pt"  # the pretrained base model
    out_path: str = "checkpoints/sft.pt"
    data_path: Optional[str] = None  # JSONL with "prompt" and "response" fields; None = synthetic task examples
    corpus_path: str = "data/input.txt"  # text the synthetic tasks are built from
    tasks: List[str] = field(default_factory=lambda: list(ALL_TASKS))
    reasoning: bool = False  # math tasks reply with a <|think|> scratchpad before the answer
    n_train: int = 20000  # number of synthetic training examples
    n_val: int = 1000  # number of synthetic held-out examples for the val loss
    val_fraction: float = 0.05  # part of data_path held out for the val loss
    batch_size: int = 64
    max_iters: int = 2000
    eval_interval: int = 200
    eval_per_task: int = 50  # held-out examples per evaluated task for measuring accuracy (0 = skip)
    max_new_tokens: int = 64  # reply length limit when evaluating (reasoning replies need ~100)
    patience: int = 3  # stop after this many evaluations without val loss improvement (0 = never stop early)
    dropout: Optional[float] = None  # None keeps the base model's dropout rate


def load_sft_data(cfg, suite):
    """ (train, val) lists of (prompt, response) """
    if cfg.data_path:
        rows = read_jsonl(cfg.data_path)
        random.Random(cfg.seed).shuffle(rows)
        n_val = max(1, int(len(rows) * cfg.val_fraction))
        pairs = [(r['prompt'], r['response']) for r in rows]
        return pairs[n_val:], pairs[:n_val]
    train = suite.sample(cfg.n_train, cfg.tasks, 'train', seed=cfg.seed)
    val = suite.sample(cfg.n_val, cfg.tasks, 'eval', seed=cfg.seed + 1)
    response = lambda e: format_response(e.answer, e.reasoning if cfg.reasoning else '')
    return [(e.prompt, response(e)) for e in train], [(e.prompt, response(e)) for e in val]


def add_reasoning_tokens(model, tokenizer):
    """ give a model the <|think|> and <|/think|> tokens (a no-op if it has them); returns how many were added """
    added = tokenizer.add_special_tokens(REASONING_TOKENS)
    if added:
        model.resize_vocab(tokenizer.vocab_size)
        print(f"added the reasoning tokens: vocab {tokenizer.vocab_size - added} -> {tokenizer.vocab_size}")
    return added


def encode_examples(tokenizer, pairs, block_size):
    encoded = [encode_chat_example(tokenizer, p, r) for p, r in pairs]
    # the whole conversation, <|end|> included, must fit in the context for the model to generate it
    kept = [e for e in encoded if len(e[0]) + 1 <= block_size]
    if len(kept) < len(encoded):
        print(f"skipped {len(encoded) - len(kept)} examples longer than block_size {block_size}")
    return kept


@torch.no_grad()
def dataset_loss(model, examples, batch_size, pad_id, device):
    """ mean loss per response token over a whole dataset """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = pad_batch(examples[i:i + batch_size], pad_id, device)
        n_tokens = (y != -100).sum().item()
        total_loss += model(x, y)[1].item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def sft(cfg):
    """ fine-tunes cfg.init_from on chat examples, keeps the checkpoint with the lowest val loss,
    returns (model, tokenizer) """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    done = skip_if_finished(cfg, device)
    if done:
        return done
    model, tokenizer, _ = load_checkpoint(cfg.init_from, device, dropout=cfg.dropout)
    suite = TaskSuite(load_text(cfg.corpus_path))
    train_pairs, val_pairs = load_sft_data(cfg, suite)
    if cfg.reasoning or any(t in r for _, r in train_pairs + val_pairs for t in REASONING_TOKENS):
        add_reasoning_tokens(model, tokenizer)
    train = encode_examples(tokenizer, train_pairs, model.config.block_size)
    val = encode_examples(tokenizer, val_pairs, model.config.block_size)
    if not train or not val:
        raise ValueError("no training or validation examples fit in the model's block_size")
    longest = max(len(t) - t.count(-100) for _, t in train)  # reply tokens, <|end|> included
    if longest > cfg.max_new_tokens:
        print(f"note: replies are up to {longest} tokens but max_new_tokens is {cfg.max_new_tokens}, so the "
              f"accuracy undercounts long replies")
    eval_set = suite.eval_set(cfg.eval_per_task, eval_tasks(cfg.tasks)) if cfg.eval_per_task else []
    print(f"device {device} | {len(train)} train / {len(val)} val examples | {len(eval_set)} eval prompts")

    rng = random.Random(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    scaler = setup_precision(cfg, device, model)
    best_val_loss, evals_without_improvement, train_losses = float('inf'), 0, []
    start_iter, state = (load_state(cfg.out_path, model, optimizer, device, cfg, scaler, rng=rng)
                         if cfg.resume else (0, None))
    if state is not None:
        best_val_loss, evals_without_improvement = state['best_val_loss'], state['evals_without_improvement']
    logger = MetricsLogger(metrics_path(cfg.out_path), append=state is not None)
    for it in range(start_iter, cfg.max_iters + 1):  # the last iteration only evaluates the final update
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters) and not (state is not None and it == start_iter):
            metrics = {'val_loss': dataset_loss(model, val, cfg.batch_size, tokenizer.pad_id, device)}
            if train_losses:
                metrics['train_loss'] = sum(train_losses) / len(train_losses)
                train_losses = []
            if eval_set:
                metrics.update(evaluate_tasks(model, tokenizer, eval_set, cfg.max_new_tokens))
            logger.log(it, **metrics)
            if metrics['val_loss'] < best_val_loss:
                best_val_loss, evals_without_improvement = metrics['val_loss'], 0
                save_checkpoint(cfg.out_path, model, tokenizer,
                                {'stage': 'sft', 'iter': it, **metrics, 'config': to_dict(cfg)})
            else:
                evals_without_improvement += 1
                if cfg.patience and evals_without_improvement >= cfg.patience:
                    print(f"early stopping: val loss has not improved for {cfg.patience} evaluations")
                    break
            if (it // cfg.eval_interval) % cfg.state_every == 0:
                save_state(cfg.out_path, model, optimizer, it, cfg, scaler, rng=rng, best_val_loss=best_val_loss,
                           evals_without_improvement=evals_without_improvement)
        if it == cfg.max_iters:
            break

        x, y = pad_batch([rng.choice(train) for _ in range(cfg.batch_size)], tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer_step(model, optimizer, loss, it, cfg, scaler)
        train_losses.append(loss.item())

    mark_finished(cfg.out_path)
    clear_state(cfg.out_path)
    model, tokenizer, _ = load_checkpoint(cfg.out_path, device)
    print(f"saved best model (val loss {best_val_loss:.4f}) to {cfg.out_path}")
    show = suite.sample(8, cfg.tasks, 'eval', seed=cfg.seed + 2)
    for example, reply in zip(show, sample_replies(model, tokenizer, [e.prompt for e in show], temperature=0.0,
                                                   max_new_tokens=cfg.max_new_tokens)):
        print(f"  {example.prompt!r} -> {reply[0][1]!r}  (expected {example.answer!r})")
    return model, tokenizer

"""Stage 2: supervised fine-tuning (SFT) turns the base model into an instruction-following chat model.

The model is trained on (prompt, response) pairs in the chat template, with the loss computed only on the
response tokens: it learns to answer, not to predict the user's message.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .chat import encode_chat_example, sample_replies
from .checkpoint import load_checkpoint, save_checkpoint
from .config import to_dict
from .data import load_text, pad_batch, read_jsonl
from .evaluation import evaluate_tasks
from .tasks import ALL_TASKS, VERIFIABLE_TASKS, TaskSuite
from .training import (MetricsLogger, TrainConfig, make_optimizer, metrics_path, optimizer_step, resolve_device,
                       set_seed)


@dataclass
class SFTConfig(TrainConfig):
    init_from: str = "checkpoints/base.pt"  # the pretrained base model
    out_path: str = "checkpoints/sft.pt"
    data_path: Optional[str] = None  # JSONL with "prompt" and "response" fields; None = synthetic task examples
    corpus_path: str = "data/input.txt"  # text the synthetic tasks are built from
    tasks: List[str] = field(default_factory=lambda: list(ALL_TASKS))
    n_train: int = 20000  # number of synthetic training examples
    n_val: int = 1000  # number of synthetic held-out examples for the val loss
    val_fraction: float = 0.05  # part of data_path held out for the val loss
    batch_size: int = 64
    max_iters: int = 2000
    eval_interval: int = 200
    eval_per_task: int = 50  # held-out examples per verifiable task for measuring accuracy (0 = skip)
    max_new_tokens: int = 64
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
    return [(e.prompt, e.answer) for e in train], [(e.prompt, e.answer) for e in val]


def encode_examples(tokenizer, pairs, block_size):
    encoded = [encode_chat_example(tokenizer, p, r) for p, r in pairs]
    kept = [e for e in encoded if len(e[0]) <= block_size]
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
    model, tokenizer, _ = load_checkpoint(cfg.init_from, device, dropout=cfg.dropout)
    suite = TaskSuite(load_text(cfg.corpus_path))
    train_pairs, val_pairs = load_sft_data(cfg, suite)
    train = encode_examples(tokenizer, train_pairs, model.config.block_size)
    val = encode_examples(tokenizer, val_pairs, model.config.block_size)
    eval_tasks = [t for t in cfg.tasks if t in VERIFIABLE_TASKS]
    eval_set = suite.eval_set(cfg.eval_per_task, eval_tasks) if cfg.eval_per_task and eval_tasks else []
    print(f"device {device} | {len(train)} train / {len(val)} val examples | {len(eval_set)} eval prompts")

    rng = random.Random(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    logger = MetricsLogger(metrics_path(cfg.out_path))
    best_val_loss, evals_without_improvement, train_losses = float('inf'), 0, []
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
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

        x, y = pad_batch([rng.choice(train) for _ in range(cfg.batch_size)], tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer_step(model, optimizer, loss, it, cfg)
        train_losses.append(loss.item())

    model, tokenizer, _ = load_checkpoint(cfg.out_path, device)
    print(f"saved best model (val loss {best_val_loss:.4f}) to {cfg.out_path}")
    show = suite.sample(8, cfg.tasks, 'eval', seed=cfg.seed + 2)
    for example, reply in zip(show, sample_replies(model, tokenizer, [e.prompt for e in show], temperature=0.0,
                                                   max_new_tokens=cfg.max_new_tokens)):
        print(f"  {example.prompt!r} -> {reply[0][1]!r}  (expected {example.answer!r})")
    return model, tokenizer

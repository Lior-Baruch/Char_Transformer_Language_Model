"""Pieces shared by every training stage: config fields, device, optimizer, learning-rate schedule, logging."""
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    """ options every training stage has; each stage's config adds its own """
    out_path: str = "checkpoints/model.pt"  # where the trained model is saved
    max_iters: int = 1000  # number of optimizer steps
    learning_rate: float = 3e-4  # peak learning rate
    min_lr: float = 3e-5  # learning rate at the end of the cosine decay
    warmup_iters: int = 100  # linear warmup from 0 to learning_rate
    weight_decay: float = 0.1
    grad_clip: float = 1.0  # max gradient norm, 0 disables clipping
    eval_interval: int = 250  # how often to evaluate
    seed: int = 1337
    device: str = "auto"  # auto, cpu, cuda or mps


def resolve_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_lr(it, cfg):
    """ linear warmup, then cosine decay from learning_rate to min_lr over the remaining iterations """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))) * (cfg.learning_rate - cfg.min_lr)


def make_optimizer(model, cfg):
    """ AdamW that applies weight decay to weight matrices and embeddings only, not to biases or LayerNorm """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': cfg.weight_decay},
        {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.99))


def optimizer_step(model, optimizer, loss, it, cfg):
    """ backprop loss and update the model with the scheduled learning rate; returns the learning rate used """
    lr = get_lr(it, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return lr


def metrics_path(out_path):
    """ checkpoints/sft.pt -> checkpoints/sft.metrics.jsonl """
    return os.path.splitext(out_path)[0] + '.metrics.jsonl'


class MetricsLogger:
    """ prints metrics and appends them as JSON lines to a file, for plotting and comparing runs """

    def __init__(self, path=None, append=False):
        self.path = path
        self.start = time.time()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if not append:
                open(path, 'w').close()

    def log(self, step, **metrics):
        metrics = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        elapsed = time.time() - self.start
        parts = [f"step {step}"] + [f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()]
        print(' | '.join(parts) + f" | {elapsed:.0f}s", flush=True)
        if self.path:
            with open(self.path, 'a') as f:
                f.write(json.dumps({'step': step, **metrics, 'time': round(elapsed, 1)}) + '\n')

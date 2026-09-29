"""Stage 3b: Group Relative Policy Optimization (GRPO), reinforcement learning with a verifiable reward.

Each step samples a group of replies to every prompt and scores them (1 if correct, 0 if not). A reply's
advantage is its reward relative to the rest of its group: (reward - group mean) / group std. There is no
learned value function. The policy is updated with the PPO clipped objective plus a KL penalty that keeps it
close to a frozen reference model:

    loss = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + kl_coef * KL(policy || reference)

where ratio = pi(token) / pi_old(token), averaged over the reply tokens of each sequence.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .chat import format_prompt, sample_replies
from .checkpoint import load_checkpoint, save_checkpoint
from .config import to_dict
from .data import load_text, pad_batch
from .evaluation import evaluate_tasks
from .model import token_logprobs
from .tasks import VERIFIABLE_TASKS, TaskSuite, score
from .training import (MetricsLogger, TrainConfig, clear_state, load_state, make_optimizer, metrics_path,
                       optimizer_step, resolve_device, save_state, set_seed)


@dataclass
class GRPOConfig(TrainConfig):
    init_from: str = "checkpoints/sft.pt"  # the model to improve (the policy)
    ref_from: Optional[str] = None  # the frozen reference model for the KL penalty; None = same as init_from
    out_path: str = "checkpoints/grpo.pt"
    corpus_path: str = "data/input.txt"
    tasks: List[str] = field(default_factory=lambda: list(VERIFIABLE_TASKS))
    batch_size: int = 16  # prompts per step
    group_size: int = 8  # replies sampled per prompt
    temperature: float = 1.0  # sampling temperature for the replies
    max_new_tokens: int = 32
    updates_per_batch: int = 1  # optimizer steps on each batch of samples (the ratio is 1 when this is 1)
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    max_iters: int = 300
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    warmup_iters: int = 10
    weight_decay: float = 0.0
    eval_interval: int = 25
    eval_per_task: int = 50  # held-out examples per verifiable task for measuring accuracy (0 = skip)
    dropout: Optional[float] = 0.0  # dropout would make the probability ratios noisy


def group_advantages(rewards, eps=1e-4):
    """ rewards (n_groups, group_size) -> advantages normalized within each group """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def grpo_loss(logps, old_logps, ref_logps, advantages, mask, clip_eps, kl_coef):
    """ per-token log-probs (N, T) under the policy, the sampling policy and the reference; advantages (N,);
    mask (N, T) marks reply tokens. Returns the loss and the mean KL to the reference """
    ratio = torch.exp(logps - old_logps)
    adv = advantages[:, None]
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # k3 estimator of KL(policy || reference): always >= 0 and unbiased
    log_ratio_ref = ref_logps - logps
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    per_token = (policy_loss + kl_coef * kl) * mask
    n_tokens = mask.sum(dim=1).clamp_min(1)
    loss = (per_token.sum(dim=1) / n_tokens).mean()
    mean_kl = ((kl * mask).sum(dim=1) / n_tokens).mean()
    return loss, mean_kl


def grpo(cfg):
    """ trains cfg.init_from with GRPO, saves the final model, returns (model, tokenizer) """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model, tokenizer, _ = load_checkpoint(cfg.init_from, device, dropout=cfg.dropout)
    ref_model, _, _ = load_checkpoint(cfg.ref_from or cfg.init_from, device, dropout=0.0)
    ref_model.eval().requires_grad_(False)
    suite = TaskSuite(load_text(cfg.corpus_path))
    eval_set = suite.eval_set(cfg.eval_per_task) if cfg.eval_per_task else []  # all tasks, not just trained ones
    print(f"device {device} | {cfg.batch_size} prompts x {cfg.group_size} replies per step | "
          f"{len(eval_set)} eval prompts")

    rng = random.Random(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    start_iter, state = (load_state(cfg.out_path, model, optimizer, device, to_dict(cfg), rng=rng)
                         if cfg.resume else (0, None))
    logger = MetricsLogger(metrics_path(cfg.out_path), append=state is not None)
    train_stats, metrics = [], {}
    for it in range(start_iter, cfg.max_iters + 1):
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters) and not (state is not None and it == start_iter):
            metrics = {}
            if train_stats:
                for k in train_stats[0]:
                    metrics[k] = sum(s[k] for s in train_stats) / len(train_stats)
                train_stats = []
            if eval_set:
                metrics.update(evaluate_tasks(model, tokenizer, eval_set, cfg.max_new_tokens))
            logger.log(it, **metrics)
            save_state(cfg.out_path, model, optimizer, it, to_dict(cfg), rng=rng)
        if it == cfg.max_iters:
            break

        # 1. sample a group of replies for each prompt and score them
        examples = [suite.make(rng.choice(cfg.tasks), rng, 'train') for _ in range(cfg.batch_size)]
        groups = sample_replies(model, tokenizer, [e.prompt for e in examples], cfg.group_size,
                                cfg.max_new_tokens, cfg.temperature)
        rewards = torch.tensor([[score(e, text) for _, text in group] for e, group in zip(examples, groups)])
        advantages = group_advantages(rewards).flatten().to(device)

        # 2. turn prompt + reply into training sequences where only the reply tokens are targets
        sequences = []
        for example, group in zip(examples, groups):
            prompt_ids = format_prompt(tokenizer, example.prompt)
            for reply_ids, _ in group:
                full = prompt_ids + reply_ids
                sequences.append((full[:-1], [-100] * (len(prompt_ids) - 1) + full[len(prompt_ids):]))
        x, y = pad_batch(sequences, tokenizer.pad_id, device)
        mask = (y != -100).float()
        with torch.no_grad():
            old_logps = token_logprobs(model, x, y)
            ref_logps = token_logprobs(ref_model, x, y)

        # 3. policy update
        for _ in range(cfg.updates_per_batch):
            logps = token_logprobs(model, x, y)
            loss, kl = grpo_loss(logps, old_logps, ref_logps, advantages, mask, cfg.clip_eps, cfg.kl_coef)
            optimizer_step(model, optimizer, loss, it, cfg)
        train_stats.append({'reward': rewards.mean().item(), 'kl': kl.item(), 'loss': loss.item(),
                            'reply_len': mask.sum(dim=1).mean().item(),
                            'groups_with_signal': (rewards.std(dim=1) > 0).float().mean().item()})

    save_checkpoint(cfg.out_path, model, tokenizer, {'stage': 'grpo', 'iter': cfg.max_iters, **metrics,
                                                     'config': to_dict(cfg)})
    clear_state(cfg.out_path)
    print(f"saved model to {cfg.out_path}")
    return model, tokenizer

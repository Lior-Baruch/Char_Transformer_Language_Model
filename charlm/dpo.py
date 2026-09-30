"""Stage 3a: Direct Preference Optimization (DPO).

DPO trains on (prompt, chosen, rejected) triples. It raises the policy's log-probability of the chosen reply
relative to a frozen reference model (usually the SFT model) and lowers it for the rejected reply:

    loss = -log sigmoid(beta * ((log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))))

beta controls how far the policy may move away from the reference. Without a preference dataset, pairs are
built from the model's own mistakes: the correct answer is "chosen" and a wrong sampled reply is "rejected".

When chosen and rejected replies are nearly identical (e.g. "136" vs "132"), DPO can lower the probability of
both, and the model gets worse. nll_coef adds the usual next-token loss on the chosen reply (as in RPO,
arXiv:2404.19733), which keeps the chosen reply likely while DPO pushes the rejected one down.
"""
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.nn import functional as F

from .chat import encode_chat_example, sample_replies
from .checkpoint import load_checkpoint, save_checkpoint
from .config import to_dict
from .data import load_text, pad_batch, read_jsonl, write_jsonl
from .evaluation import evaluate_tasks
from .model import token_logprobs
from .reasoning import format_response
from .tasks import VERIFIABLE_TASKS, TaskSuite, eval_tasks, score
from .training import (MetricsLogger, TrainConfig, clear_state, load_state, make_optimizer, metrics_path,
                       optimizer_step, resolve_device, save_state, set_seed, setup_precision, skip_if_finished,
                       state_path)


@dataclass
class DPOConfig(TrainConfig):
    init_from: str = "checkpoints/sft.pt"  # the model to improve (the policy)
    ref_from: Optional[str] = None  # the frozen reference model; None = same as init_from
    out_path: str = "checkpoints/dpo.pt"
    data_path: Optional[str] = None  # JSONL with prompt/chosen/rejected; None = build pairs from the model's mistakes
    corpus_path: str = "data/input.txt"
    tasks: List[str] = field(default_factory=lambda: list(VERIFIABLE_TASKS))
    # whether the chosen replies to math tasks include the step-by-step reasoning; None = when the model has the
    # reasoning tokens (i.e. it was fine-tuned with reasoning=true)
    reasoning: Optional[bool] = None
    n_pairs: int = 4000  # pairs to build when data_path is None
    samples_per_prompt: int = 4  # replies sampled per prompt when looking for a wrong one
    sample_temperature: float = 1.0
    val_fraction: float = 0.05  # part of the pairs held out for the val loss
    beta: float = 0.1
    nll_coef: float = 0.0  # weight of an extra next-token loss on the chosen replies (0 = plain DPO)
    batch_size: int = 32  # pairs per step
    max_iters: int = 1000
    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    warmup_iters: int = 20
    weight_decay: float = 0.0
    eval_interval: int = 100
    eval_per_task: int = 50  # held-out examples per evaluated task for measuring accuracy (0 = skip)
    max_new_tokens: int = 32  # reply length limit when sampling and evaluating (reasoning replies need ~100)
    dropout: Optional[float] = 0.0  # dropout would make the log-probabilities noisy


def build_preference_pairs(model, tokenizer, suite, n_pairs, tasks=VERIFIABLE_TASKS, samples_per_prompt=4,
                           temperature=1.0, max_new_tokens=32, seed=0, prompts_per_round=512, max_rounds=50,
                           reasoning=False):
    """ sample replies to training prompts; for each prompt the model got wrong at least once, pair the correct
    reply (chosen: the answer, preceded by its reasoning trace if reasoning) with one of the wrong ones (rejected) """
    rng = random.Random(seed)
    pairs = []
    for round_ in range(max_rounds):
        examples = suite.sample(min(prompts_per_round, 4 * n_pairs), tasks, 'train', seed=seed + round_)
        replies = sample_replies(model, tokenizer, [e.prompt for e in examples], samples_per_prompt,
                                 max_new_tokens, temperature)
        for example, group in zip(examples, replies):
            wrong = [text for _, text in group if score(example, text) == 0.0]
            if wrong:
                chosen = format_response(example.answer, example.reasoning if reasoning else '')
                pairs.append({'task': example.task, 'prompt': example.prompt,
                              'chosen': chosen, 'rejected': rng.choice(wrong)})
                if len(pairs) == n_pairs:
                    return pairs
    print(f"found only {len(pairs)} pairs: the model rarely makes mistakes on these tasks")
    return pairs


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """ DPO loss from summed log-probabilities of each reply, all (B,); returns the loss and some statistics """
    chosen_rewards = beta * (policy_chosen - ref_chosen)  # the "implicit rewards" of DPO
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    margins = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margins).mean()
    stats = {'reward_acc': (margins > 0).float().mean().item(), 'margin': margins.mean().item(),
             'chosen_reward': chosen_rewards.mean().item(), 'rejected_reward': rejected_rewards.mean().item()}
    return loss, stats


def sequence_logprobs(model, sequences, pad_id, device):
    """ summed log-probability of the target (reply) tokens of each (inputs, targets) sequence, (N,) """
    x, y = pad_batch(sequences, pad_id, device)
    return token_logprobs(model, x, y).sum(-1)


def pair_logprobs(model, pairs, pad_id, device):
    """ log-probabilities of the chosen and rejected replies, computed in one batch """
    logps = sequence_logprobs(model, [c for c, _ in pairs] + [r for _, r in pairs], pad_id, device)
    return logps[:len(pairs)], logps[len(pairs):]


@torch.no_grad()
def reference_logprobs(ref_model, pairs, pad_id, device, batch_size=64):
    """ the reference model never changes, so its log-probabilities are computed once up front """
    chosen, rejected = [], []
    for i in range(0, len(pairs), batch_size):
        c, r = pair_logprobs(ref_model, pairs[i:i + batch_size], pad_id, device)
        chosen.append(c)
        rejected.append(r)
    return torch.cat(chosen), torch.cat(rejected)


@torch.no_grad()
def validate(model, pairs, ref_chosen, ref_rejected, cfg, pad_id, device):
    model.eval()
    total, stats_sum = 0.0, {}
    for i in range(0, len(pairs), cfg.batch_size):
        c, r = pair_logprobs(model, pairs[i:i + cfg.batch_size], pad_id, device)
        loss, stats = dpo_loss(c, r, ref_chosen[i:i + cfg.batch_size], ref_rejected[i:i + cfg.batch_size], cfg.beta)
        n = len(c)
        total += loss.item() * n
        for k, v in stats.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v * n
    model.train()
    return {'val_loss': total / len(pairs), 'val_reward_acc': stats_sum['reward_acc'] / len(pairs)}


def dpo(cfg):
    """ trains cfg.init_from with DPO, saves the final model, returns (model, tokenizer) """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    done = skip_if_finished(cfg, device)
    if done:
        return done
    model, tokenizer, _ = load_checkpoint(cfg.init_from, device, dropout=cfg.dropout)
    ref_model, ref_tokenizer, _ = load_checkpoint(cfg.ref_from or cfg.init_from, device, dropout=0.0)
    if ref_tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("the reference model's tokenizer differs from the policy's, so their "
                         "log-probabilities can't be compared")
    ref_model.eval().requires_grad_(False)
    scaler = setup_precision(cfg, device, model, ref_model)
    reasoning = tokenizer.has_reasoning_tokens if cfg.reasoning is None else cfg.reasoning
    suite = TaskSuite(load_text(cfg.corpus_path))

    pairs_path = os.path.splitext(cfg.out_path)[0] + '.pairs.jsonl'
    resuming = cfg.resume and os.path.exists(state_path(cfg.out_path))
    if cfg.data_path:
        rows = read_jsonl(cfg.data_path)
    elif resuming and os.path.exists(pairs_path):
        rows = read_jsonl(pairs_path)  # the pairs built before the interruption
    else:
        print(f"building {cfg.n_pairs} preference pairs from {cfg.init_from}'s own mistakes...")
        rows = build_preference_pairs(model, tokenizer, suite, cfg.n_pairs, cfg.tasks, cfg.samples_per_prompt,
                                      cfg.sample_temperature, cfg.max_new_tokens, seed=cfg.seed, reasoning=reasoning)
        write_jsonl(pairs_path, rows)
        print(f"saved {len(rows)} pairs to {pairs_path}")
    random.Random(cfg.seed).shuffle(rows)
    pairs = [(encode_chat_example(tokenizer, r['prompt'], r['chosen']),
              encode_chat_example(tokenizer, r['prompt'], r['rejected'])) for r in rows]
    pairs = [p for p in pairs if max(len(p[0][0]), len(p[1][0])) <= model.config.block_size]
    if len(pairs) < 2:
        raise ValueError(f"need at least 2 preference pairs to train and validate, got {len(pairs)}")
    n_val = max(1, int(len(pairs) * cfg.val_fraction))
    val, train = pairs[:n_val], pairs[n_val:]
    ref_chosen, ref_rejected = reference_logprobs(ref_model, pairs, tokenizer.pad_id, device)
    val_ref, train_ref = (ref_chosen[:n_val], ref_rejected[:n_val]), (ref_chosen[n_val:], ref_rejected[n_val:])
    eval_set = suite.eval_set(cfg.eval_per_task, eval_tasks(cfg.tasks)) if cfg.eval_per_task else []
    print(f"device {device} | {len(train)} train / {len(val)} val pairs | {len(eval_set)} eval prompts")

    rng = random.Random(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    start_iter, state = (load_state(cfg.out_path, model, optimizer, device, cfg, scaler, rng=rng)
                         if cfg.resume else (0, None))
    logger = MetricsLogger(metrics_path(cfg.out_path), append=state is not None)
    train_stats, metrics = [], {}
    for it in range(start_iter, cfg.max_iters + 1):
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters) and not (state is not None and it == start_iter):
            metrics = validate(model, val, *val_ref, cfg, tokenizer.pad_id, device)
            if train_stats:
                for k in train_stats[0]:
                    metrics[f'train_{k}'] = sum(s[k] for s in train_stats) / len(train_stats)
                train_stats = []
            if eval_set:
                metrics.update(evaluate_tasks(model, tokenizer, eval_set, cfg.max_new_tokens))
            logger.log(it, **metrics)
            if (it // cfg.eval_interval) % cfg.state_every == 0:
                save_state(cfg.out_path, model, optimizer, it, cfg, scaler, rng=rng)
        if it == cfg.max_iters:
            break

        idx = [rng.randrange(len(train)) for _ in range(cfg.batch_size)]
        batch = [train[i] for i in idx]
        chosen, rejected = pair_logprobs(model, batch, tokenizer.pad_id, device)
        loss, stats = dpo_loss(chosen, rejected, train_ref[0][idx], train_ref[1][idx], cfg.beta)
        if cfg.nll_coef:
            n_tokens = torch.tensor([sum(t != -100 for t in c[1]) for c, _ in batch], device=device)
            loss = loss + cfg.nll_coef * (-chosen / n_tokens).mean()  # mean next-token loss of the chosen replies
        optimizer_step(model, optimizer, loss, it, cfg, scaler)
        train_stats.append({'loss': loss.item(), 'reward_acc': stats['reward_acc'],
                            'chosen_reward': stats['chosen_reward'], 'rejected_reward': stats['rejected_reward']})

    save_checkpoint(cfg.out_path, model, tokenizer, {'stage': 'dpo', 'iter': cfg.max_iters, **metrics,
                                                     'config': to_dict(cfg), 'finished': True})
    clear_state(cfg.out_path)
    print(f"saved model to {cfg.out_path}")
    return model, tokenizer

"""Pieces shared by every training stage: config fields, device, precision, optimizer, learning-rate schedule,
resuming and logging."""
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch

from .checkpoint import atomic_save, is_finished, load_checkpoint
from .config import to_dict


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
    # fp32, bf16, fp16 (with loss scaling) or auto: bf16 on GPUs that support it natively (A100, L4, ...),
    # fp16 on older GPUs (T4, V100), fp32 on CPU. 16-bit training is 2-8x faster on a GPU
    precision: str = "fp32"
    resume: bool = False  # continue an interrupted run from its last saved state (*.state.pt); skips finished runs
    state_every: int = 1  # save the resume state every this many evaluations (a large model's state is big)


def resolve_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


AMP_DTYPES = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def resolve_precision(precision, device):
    """ 'auto' -> the fastest precision the device handles well """
    if precision == 'auto':
        if device.startswith('cuda') and torch.cuda.is_available():
            # compute capability 8+ (A100, L4, RTX 30xx...) has native bf16; older GPUs only emulate it slowly
            return 'bf16' if torch.cuda.get_device_capability()[0] >= 8 else 'fp16'
        return 'fp32'
    if precision not in AMP_DTYPES:
        raise ValueError(f"precision must be auto, {', '.join(AMP_DTYPES)}; got {precision!r}")
    return precision


def setup_precision(cfg, device, *models):
    """ resolve cfg.precision, make the models' forward passes run in it, and return the GradScaler (enabled only
    for fp16, whose small range needs the loss scaled up so small gradients don't round to zero) """
    precision = resolve_precision(cfg.precision, device)
    for model in models:
        model.autocast_dtype = AMP_DTYPES[precision]
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True  # faster float32 matrix multiplies on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True
    print(f"precision {precision}")
    enabled = precision == 'fp16'
    if hasattr(torch.amp, 'GradScaler'):  # PyTorch 2.3+
        return torch.amp.GradScaler(device.split(':')[0], enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and device.startswith('cuda'))


def skip_if_finished(cfg, device):
    """ with resume=true, a stage whose checkpoint is complete (and has no state left to resume) is not run again,
    so re-running every cell of a notebook after a disconnect never overwrites finished work.
    returns (model, tokenizer) of the finished checkpoint, or None when the stage should run """
    if cfg.resume and not os.path.exists(state_path(cfg.out_path)) and is_finished(cfg.out_path):
        print(f"{cfg.out_path} is already finished; skipping (delete it, or set resume=false, to train it again)")
        model, tokenizer, _ = load_checkpoint(cfg.out_path, device)
        return model, tokenizer
    return None


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


def optimizer_step(model, optimizer, loss, it, cfg, scaler=None):
    """ backprop loss and update the model with the scheduled learning rate; returns the learning rate used.
    With an enabled GradScaler (fp16), the loss is scaled for backprop and the gradients unscaled before clipping """
    lr = get_lr(it, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)  # skips the step if the gradients overflowed
        scaler.update()
        return lr
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return lr


def state_path(out_path):
    """ checkpoints/sft.pt -> checkpoints/sft.state.pt """
    return os.path.splitext(out_path)[0] + '.state.pt'


def _precision_of(model):
    return next(name for name, dtype in AMP_DTYPES.items() if dtype == model.autocast_dtype)


def save_state(out_path, model, optimizer, it, cfg, scaler=None, **extra):
    """ everything needed to continue a run from iteration it: weights, optimizer, loss scaler, random-number
    generators, the run's config and precision, and any stage-specific values (a random.Random in extra is stored
    by its state) """
    extra = {k: ('py_rng', v.getstate()) if isinstance(v, random.Random) else v for k, v in extra.items()}
    atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _get_rng_states(),
                 'iter': it, 'config': to_dict(cfg), 'precision': _precision_of(model),
                 'scaler': scaler.state_dict() if scaler is not None and scaler.is_enabled() else {},
                 'extra': extra}, state_path(out_path))


def _mps_available():
    return getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()


def _get_rng_states():
    """ torch's random-number generators: the CPU one and, when present, the GPU ones (dropout and sampling on a
    GPU draw from those) """
    states = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    if _mps_available():
        states['mps'] = torch.mps.get_rng_state()
    return states


def _set_rng_states(states):
    torch.set_rng_state(states['cpu'].cpu())
    restored = {'cpu'}
    if 'cuda' in states and torch.cuda.is_available() and len(states['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in states['cuda']])
        restored.add('cuda')
    if 'mps' in states and _mps_available():
        torch.mps.set_rng_state(states['mps'].cpu())
        restored.add('mps')
    if restored != set(_get_rng_states()) or restored != set(states):
        print("note: resuming on different hardware than the run was saved on, so random draws (dropout, "
              "sampling) will differ from an uninterrupted run")


def load_state(out_path, model, optimizer, device, cfg, scaler=None, **rngs):
    """ restore a run saved by save_state; random.Random objects passed in rngs get their saved state back.
    returns (iteration to continue from, the other extra values), or (0, None) when there is nothing to resume.
    Refuses to resume when the settings differ from the saved run's, since that would mix two different runs
    (a setting added to charlm after the state was saved counts as its default) """
    path = state_path(out_path)
    if not os.path.exists(path):
        return 0, None
    state = torch.load(path, map_location=device, weights_only=True)
    config, defaults = to_dict(cfg), to_dict(type(cfg)())
    ignored = {'resume', 'device', 'precision', 'state_every'}  # the precision is compared once resolved, below
    changed = sorted(k for k in set(config) | set(state['config']) if k not in ignored
                     and config.get(k, defaults.get(k)) != state['config'].get(k, defaults.get(k)))
    saved_precision = state.get('precision', 'fp32')
    if saved_precision != _precision_of(model):
        changed.append(f"precision {saved_precision} vs {_precision_of(model)}")
    if changed:
        raise ValueError(f"{path} was saved by a run with different settings ({', '.join(changed)}); "
                         f"restore those settings to resume it, or delete the file to start over")
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and scaler.is_enabled() and state.get('scaler'):
        scaler.load_state_dict(state['scaler'])
    _set_rng_states(state['rng'])
    extra = {}
    for k, v in state['extra'].items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == 'py_rng':
            rngs[k].setstate(_to_tuple(v[1]))
        else:
            extra[k] = v
    print(f"resuming from iteration {state['iter']}")
    return state['iter'], extra


def _to_tuple(x):
    return tuple(_to_tuple(i) for i in x) if isinstance(x, (tuple, list)) else x


def clear_state(out_path):
    """ a finished run has nothing left to resume """
    if os.path.exists(state_path(out_path)):
        os.remove(state_path(out_path))


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

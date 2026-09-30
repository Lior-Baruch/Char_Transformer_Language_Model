"""Saving and loading a model together with its config and tokenizer."""
import dataclasses
import os

import torch

from .model import CharTransformerLanguageModel, ModelConfig
from .tokenizer import CharTokenizer


def atomic_save(obj, path):
    """ torch.save to a temporary file, then rename it into place, so a crash mid-write (a Colab disconnect, a full
    disk) never leaves a half-written file behind: the path holds either the old or the new version """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, tokenizer, meta=None):
    """ one file holds everything needed to rebuild the model: config, tokenizer, weights and free-form metadata """
    atomic_save({
        'model_config': dataclasses.asdict(model.config),
        'tokenizer': tokenizer.to_dict(),
        'state_dict': model.state_dict(),
        'meta': meta or {},
    }, path)


def load_checkpoint(path, device='cpu', dropout=None):
    """ returns (model, tokenizer, meta); dropout overrides the saved dropout rate (e.g. 0.0 for fine-tuning) """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**ckpt['model_config'])
    if dropout is not None:
        config.dropout = dropout
    model = CharTransformerLanguageModel(config)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    tokenizer = CharTokenizer.from_dict(ckpt['tokenizer'])
    assert tokenizer.vocab_size == config.vocab_size, f"{path}: the tokenizer and the model disagree on the vocabulary"
    return model, tokenizer, ckpt['meta']


def mark_finished(path):
    """ record in a checkpoint's metadata that its training run completed """
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    ckpt['meta']['finished'] = True
    atomic_save(ckpt, path)


def is_finished(path):
    """ whether path is a checkpoint whose training run completed """
    return (os.path.exists(path)
            and torch.load(path, map_location='cpu', weights_only=True)['meta'].get('finished', False))

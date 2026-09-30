"""Saving and loading a model together with its config and tokenizer."""
import dataclasses
import os

import torch

from .model import CharTransformerLanguageModel, ModelConfig
from .tokenizer import CharTokenizer


def save_checkpoint(path, model, tokenizer, meta=None):
    """ one file holds everything needed to rebuild the model: config, tokenizer, weights and free-form metadata """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
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
    return model, CharTokenizer.from_dict(ckpt['tokenizer']), ckpt['meta']

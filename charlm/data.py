"""Loading text and JSONL data, and building training batches."""
import json
import os

import torch


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def get_text_batch(data, batch_size, block_size, device, generator=None):
    """ random windows of a long 1-D token tensor: inputs x and targets y (x shifted by one), both (B, T) """
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def pad_batch(sequences, pad_id, device):
    """ right-pad (inputs, targets) pairs of token lists into (B, T) tensors; padded targets are -100 (no loss) """
    T = max(len(x) for x, _ in sequences)
    x = torch.full((len(sequences), T), pad_id, dtype=torch.long)
    y = torch.full((len(sequences), T), -100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(sequences):
        x[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return x.to(device), y.to(device)

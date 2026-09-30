"""Loading text and JSONL data, and building training batches."""
import glob
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


def expand_paths(data_path):
    """ the text files a data_path option names: a file, a directory (the .txt files in it), a glob pattern such
    as "data/*.txt", or a list of these. Each directory or pattern is sorted, so the order never depends on the
    file system """
    items = [data_path] if isinstance(data_path, str) else list(data_path)
    paths = []
    for item in items:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, '*.txt')))
        elif glob.has_magic(item):
            found = sorted(glob.glob(item))
        elif os.path.exists(item):
            found = [item]
        else:
            raise FileNotFoundError(f"no such file: {item!r}")
        if not found:
            raise FileNotFoundError(f"no text files match {item!r}")
        paths += [p for p in found if p not in paths]
    return paths


def read_text_bytes(path):
    """ a text file's bytes, with \\r\\n and \\r line endings turned into \\n like reading in text mode does.
    A bytearray, so torch can use it without a copy (a 1 GB corpus then needs 1 GB of memory, not 8) """
    buf = bytearray(os.path.getsize(path))
    with open(path, 'rb') as f:
        f.readinto(buf)
    if b'\r' in buf:
        buf = buf.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    return buf


def text_chars(buf):
    """ the set of characters in a text file's bytes """
    if not buf.isascii():
        return set(buf.decode('utf-8'))
    if not buf:
        return set()
    counts = torch.bincount(torch.frombuffer(buf, dtype=torch.uint8), minlength=128)
    return {chr(i) for i in counts.nonzero().flatten().tolist()}


def encode_text_bytes(tokenizer, buf):
    """ token ids of a text file's bytes (no special tokens) as a 1-D tensor: uint8, one byte per character, when
    the vocabulary has fewer than 256 tokens, else int64. ASCII text is encoded with bytearray.translate, which is
    ~100x faster than tokenizer.encode and handles a 1 GB corpus in seconds """
    small = tokenizer.vocab_size < 256
    if small and buf.isascii():
        missing = 255  # never a real id here, since the vocabulary is smaller than 256
        ids = buf.translate(bytes(tokenizer.stoi.get(chr(i), missing) for i in range(256)))
        bad = ids.find(missing)
        if bad >= 0:
            raise ValueError(f"character {chr(buf[bad])!r} is not in the tokenizer's vocabulary")
        return torch.frombuffer(ids, dtype=torch.uint8) if ids else torch.zeros(0, dtype=torch.uint8)
    ids = tokenizer.encode(buf.decode('utf-8'), allow_special=False)
    return torch.tensor(ids, dtype=torch.uint8 if small else torch.long)


def get_text_batch(data, batch_size, block_size, device, generator=None):
    """ random windows of a long 1-D token tensor: inputs x and targets y (x shifted by one), both (B, T) int64 """
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device).long(), y.to(device).long()


def pad_batch(sequences, pad_id, device):
    """ right-pad (inputs, targets) pairs of token lists into (B, T) tensors; padded targets are -100 (no loss) """
    T = max(len(x) for x, _ in sequences)
    x = torch.full((len(sequences), T), pad_id, dtype=torch.long)
    y = torch.full((len(sequences), T), -100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(sequences):
        x[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return x.to(device), y.to(device)

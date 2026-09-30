"""The chat template and helpers to get replies from a model.

A conversation turn is laid out as

    <|user|>Reverse the word: love<|assistant|>evol<|end|>

The model is trained (SFT) to write the reply after <|assistant|> and finish it with <|end|>. A model taught to
reason first writes a scratchpad between <|think|> and <|/think|> (see reasoning.py):

    <|user|>What is 47 + 85?<|assistant|><|think|>7+5+0=12 A=2, 4+8+1=13 A=132 => 132<|/think|>132<|end|>
"""
from collections import defaultdict

import torch

from .reasoning import split_reply
from .tokenizer import REASONING_TOKENS


def format_prompt(tokenizer, prompt):
    """ token ids of the prompt part: <|user|>{prompt}<|assistant|> """
    return [tokenizer.user_id] + tokenizer.encode(prompt, allow_special=False) + [tokenizer.assistant_id]


def encode_response(tokenizer, response):
    """ token ids of a reply's text; its <|think|>/<|/think|> markers become the reasoning tokens """
    missing = [t for t in REASONING_TOKENS if t in response and t not in tokenizer.stoi]
    if missing:
        raise ValueError(f"the response contains {missing[0]} but the tokenizer has no such token; add the "
                         f"reasoning tokens first (tokenizer.add_special_tokens + model.resize_vocab, which SFT does "
                         f"with reasoning=true)")
    return tokenizer.encode(response, allow_special=REASONING_TOKENS)


def encode_chat_example(tokenizer, prompt, response):
    """ (inputs, targets) for next-token training where only the response and <|end|> are predicted;
    the prompt positions get target -100 so they add nothing to the loss """
    prompt_ids = format_prompt(tokenizer, prompt)
    full = prompt_ids + encode_response(tokenizer, response) + [tokenizer.end_id]
    inputs, targets = full[:-1], full[1:]
    targets = [-100] * (len(prompt_ids) - 1) + targets[len(prompt_ids) - 1:]
    return inputs, targets


def sample_replies(model, tokenizer, prompts, num_samples=1, max_new_tokens=64, temperature=1.0, top_k=None,
                   max_batch=256):
    """ for each prompt, num_samples replies as (reply token ids, reply text).
    The token ids include the final <|end|> if the model produced one; the text never does.
    Prompts of the same length are batched together, so no padding is needed. """
    device = next(model.parameters()).device
    encoded = [format_prompt(tokenizer, p) for p in prompts]
    by_length = defaultdict(list)
    for i, ids in enumerate(encoded):
        by_length[len(ids)].append(i)

    results = [None] * len(prompts)
    rows_per_call = max(1, max_batch // num_samples)
    for length, indices in by_length.items():
        room = model.config.block_size - length  # keep prompt + reply within the context
        if room <= 0:
            raise ValueError(f"a prompt of {length} tokens leaves no room for a reply "
                             f"within block_size {model.config.block_size}")
        new_tokens = min(max_new_tokens, room)
        for start in range(0, len(indices), rows_per_call):
            chunk = indices[start:start + rows_per_call]
            x = torch.tensor([encoded[i] for i in chunk], dtype=torch.long, device=device)
            x = x.repeat_interleave(num_samples, dim=0)
            out = model.generate(x, new_tokens, temperature=temperature, top_k=top_k,
                                 stop_token=tokenizer.end_id)[:, length:].tolist()
            for j, i in enumerate(chunk):
                results[i] = [_cut_reply(tokenizer, out[j * num_samples + s]) for s in range(num_samples)]
    return results


def _cut_reply(tokenizer, ids):
    if tokenizer.end_id in ids:
        ids = ids[:ids.index(tokenizer.end_id) + 1]
        return ids, tokenizer.decode(ids[:-1])
    return ids, tokenizer.decode(ids)


def chat(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, top_k=None, return_reasoning=False):
    """ the model's answer to one message (greedy by default). With return_reasoning, returns
    (reasoning, answer): the scratchpad it wrote between <|think|> and <|/think|> ('' if it didn't reason) and the
    final answer ('' if the reasoning was cut off before it finished) """
    reply = sample_replies(model, tokenizer, [prompt], 1, max_new_tokens, temperature, top_k)[0][0][1]
    reasoning, answer = split_reply(reply)
    return (reasoning, answer) if return_reasoning else answer

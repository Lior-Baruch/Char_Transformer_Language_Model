"""A GPT-style (decoder-only) transformer language model."""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 100  # number of tokens, also known as V (set from the tokenizer)
    block_size: int = 256  # maximum context length, also known as T
    n_embd: int = 384  # embedding dimension, also known as C
    n_head: int = 6  # number of attention heads, also known as H
    n_layer: int = 6  # number of transformer blocks, also known as L
    dropout: float = 0.2  # dropout rate


class CausalSelfAttention(nn.Module):
    """ all attention heads computed together; each position only attends to itself and earlier positions """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # queries, keys and values of every head
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # combines the heads
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # 3 x (B, T, C)
        # split the channels into heads, (B, T, C) -> (B, H, T, C/H)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(q @ k^T / sqrt(head_size)) @ v, with future positions masked out
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate the heads, (B, H, T, C/H) -> (B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """ expand to 4x the embedding size, apply a non-linearity, project back down """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feed-forward) """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # pre-norm + residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class CharTransformerLanguageModel(nn.Module):
    """ predicts the next token from the previous ones """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)  # (V, C)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)  # (T, C)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # (C, V)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """ idx: (B, T) token ids. targets: (B, T) next-token ids, -100 where no loss should be computed.
        returns logits (B, T, V) and the mean cross-entropy loss (or None without targets) """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """ extend idx (B, T) by sampling up to max_new_tokens tokens.
        temperature 0 picks the most likely token. Once a row samples stop_token, it keeps emitting stop_token,
        and generation ends early when every row has stopped. """
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])  # crop to the last block_size tokens
            logits = logits[:, -1, :]  # only the last position predicts the next token, (B, V)
            if temperature == 0:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float('-inf'))
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # (B, 1)
            if stop_token is not None:
                idx_next = idx_next.masked_fill(finished[:, None], stop_token)
                finished |= idx_next[:, 0] == stop_token
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_token is not None and finished.all():
                break
        self.train(was_training)
        return idx


def token_logprobs(model, idx, targets):
    """ log-probability the model assigns to each target token, (B, T); positions where targets is -100 get 0 """
    logits, _ = model(idx)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = targets != -100
    picked = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return picked * mask

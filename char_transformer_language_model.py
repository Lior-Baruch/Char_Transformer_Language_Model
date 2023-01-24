import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many sequences to train on at once, also known as B
block_size = 256  # how long each context sequence is, also known as T
max_iters = 1000
eval_interval = 500  # how often to evaluate the model
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
save_interval = 1000  # how often to save the model
n_embd = 384  # embedding dimension, also known as hidden size or C
n_head = 6  # number of heads, also known as H
n_layer = 6  # number of layers, also known as L
dropout = 0.2  # dropout rate, also known as D
# ------------

torch.manual_seed(1337)
print('Using device:', device)

# Load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()  # data is a long string

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}


# encode using the mapping
def encode(s):
    return [char_to_int[c] for c in s]


# decode using the mapping
def decode(l):
    return ''.join([int_to_char[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)  # convert to tensor
n = int(0.9 * len(data))  # first 90% will be trained, the rest for validation
train_data = data[:n]
val_data = data[n:]


# batching and shuffling
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # random starting points
    x = torch.stack([data[i:i + block_size] for i in ix])  # input
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # target
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # don't track gradients
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # get a batch of data
            logits, loss = model.forward(X, Y)  # forward pass
            losses[k] = loss.item()  # store the loss
        out[split] = losses.mean()  # average loss
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention
    this is the part that computes the attention scores and performs the weighted aggregation of the values """

    def __init__(self, head_size):
        super().__init__()
        # these are the three linear layers that are used to compute the attention scores
        self.key = nn.Linear(n_embd, head_size, bias=False)  # (B,T,C) -> (B,T,H)
        self.query = nn.Linear(n_embd, head_size, bias=False)  # (B,T,C) -> (B,T,H)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (B,T,C) -> (B,T,H)

        # this is the masking trick, it is used to prevent the model from attending to the future
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # this is the dropout, it is used to prevent over-fitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is the input, it has shape (B, T, C)
        B, T, C = x.shape

        # compute the keys, queries and values, (B, T, C) -> (B, T, H)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # compute attention scores ("affinities"), (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # masking trick, future is masked
        wei = F.softmax(wei, dim=-1)  # softmax to get the weights
        wei = self.dropout(wei)  # dropout, to prevent over-fitting

        # perform the weighted aggregation of the values, (B, T, T) @ (B, T, H) -> (B, T, H)
        out = wei @ v  # weighted aggregation
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # this is the list of heads
        self.proj = nn.Linear(n_embd, n_embd)  # this is the projection layer, it is used to combine the heads
        self.dropout = nn.Dropout(dropout)  # this is the dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # this is the concatenation of the heads
        out = self.proj(out)  # this is the projection, it is used to combine the heads
        out = self.dropout(out)  # this is the dropout, it is used to prevent over-fitting
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        # sim
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head  # split the embedding dimension into n_head heads
        self.sa = MultiHeadAttention(n_head, head_size)  # self-attention
        self.ffwd = FeedForward(n_embd)  # feed-forward network
        self.ln1 = nn.LayerNorm(n_embd)  # used to normalize the input of the self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # used to normalize the input of the feed-forward network

    def forward(self, x):
        x_temp = self.ln1(x)  # normalize the input
        x_temp = self.sa(x_temp)  # self-attention
        x = x + x_temp  # residual connection
        x_temp = self.ln2(x)  # normalize the input
        x_temp = self.ffwd(x_temp)  # feed-forward network
        x = x + x_temp  # residual connection
        return x


class BigramLanguageModel(nn.Module):
    """ a simple bigram language model, it is used to initialize the parameters of the transformer """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token embedding table, (V,C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # position embedding table, (T,C)
        self.TransformerBlocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])  # transformer blocks, (B,T,C)
        self.ln_f = nn.LayerNorm(n_embd)  # used to normalize the input of the final linear layer, (B,T,C)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # the final linear layer, (B,T,vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # token embedding, (B,T) -> (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # position embedding, (T) -> (T,C)
        x = tok_emb + pos_emb  # add the token and position embeddings, (B,T,C) + (T,C) -> (B,T,C)
        x = self.TransformerBlocks(x)  # transformer blocks, (B,T,C) -> (B,T,C)
        x = self.ln_f(x)  # normalize the input of the final linear layer, (B,T,C) -> (B,T,C)
        logits = self.lm_head(x)  # the final linear layer, (B,T,C) -> (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape logits and targets to (B*T, C) and (B*T,)
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # compute the loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
model_device = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model_device.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model_device.parameters(), lr=learning_rate)

# train the model
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model_device(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model_device.generate(context, max_new_tokens=500)[0].tolist()))

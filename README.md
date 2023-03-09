# Char Transformer Language Model - Bigram Language Model
This code implements a Bigram Language Model using a transformer architecture. A Bigram Language Model predicts the next character in a sequence of characters given the previous character. The transformer architecture used in this code consists of a stack of n_layer Transformer Blocks, each of which contains a multi-head self-attention mechanism followed by a feed-forward neural network.

# Classes
**Head**
One head of self-attention. Computes the attention scores and performs the weighted aggregation of the values.

**MultiHeadAttention**
Multiple heads of self-attention in parallel.

**FeedForward**
A simple linear layer followed by a non-linearity.

**TransformerBlock**
Transformer block: communication followed by computation.

**BigramLanguageModel**
A simple bigram language model, used to initialize the parameters of the transformer. Consists of the following components:

token_embedding_table: a lookup table for token embeddings.
position_embedding_table: a lookup table for position embeddings.
TransformerBlocks: a stack of n_layer Transformer Blocks.
ln_f: a layer normalization layer for the input of the final linear layer.
lm_head: the final linear layer.
# Other Functions and Variables
encode(s): encodes a string s into a list of integers using the char_to_int dictionary.
decode(l): decodes a list of integers l into a string using the int_to_char dictionary.
get_batch(split): generates a small batch of data of inputs x and targets y for a given split (train or validation).
estimate_loss(): estimates the loss on the train and validation sets using a fixed number of evaluation iterations (eval_iters).
model: an instance of the BigramLanguageModel class.
model_device: the model instance moved to the device (CPU or GPU) specified by the device variable.
optimizer: a PyTorch optimizer (AdamW) used to optimize the parameters of the model.
batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, save_interval, n_embd, n_head, n_layer, dropout: hyperparameters used in the model.
# Usage
To use this code, simply run the Python script. The script will train the model on the input text file (input.txt) and print the train and validation losses every eval_interval iterations. Once training is complete, the script will generate a sequence of characters using the trained model and print it to the console. The generated sequence can be modified by changing the max_new_tokens parameter passed to the generate function.

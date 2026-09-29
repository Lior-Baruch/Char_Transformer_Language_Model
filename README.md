# Character-Level Transformer Language Model

This repository contains an implementation of a GPT-style (decoder-only) transformer language model. This character-level language model aims to predict the next character in a sequence given the previous characters. The model is built and trained using the PyTorch library.

## Dependencies

The only dependency is [PyTorch](https://pytorch.org/).

The code is designed to utilize a GPU if available. If not, it will default to a CPU.

## Data

The model is trained on `input.txt`, a text file containing the works of Shakespeare. The text is loaded from the file as a long string. The unique characters in the text are identified, and a mapping from characters to integers (and vice versa) is created. This mapping is used to encode the text into a format that the model can understand.

The data is then split into a training set (90% of the data) and a validation set (10% of the data). The training set is used to train the model, and the validation set is used to monitor the model's performance during training.

## Model

The transformer model consists of several key components, each implemented as a separate Python class:

- **Head**: This class represents a single head of the self-attention mechanism. It computes attention scores and performs the weighted aggregation of values.

- **MultiHeadAttention**: This class manages multiple instances of the `Head` class in parallel, allowing the model to capture different types of information from the input data.

- **FeedForward**: This class represents a feed-forward network: a linear layer that expands the embedding 4x, a ReLU non-linearity, and a linear layer that projects back down.

- **TransformerBlock**: Each Transformer Block contains a self-attention mechanism followed by a feed-forward network.

- **CharTransformerLanguageModel**: This class represents the main model. It includes token and position embeddings, a series of transformer blocks, a layer normalization layer, and a final linear layer.

## Training

The model is trained using the AdamW optimizer with a learning rate of 3e-4. The training process involves forward propagation, loss computation, backpropagation, and parameter update steps. The cross-entropy loss function is used to calculate the loss between the model's predictions and the actual characters. The training and validation losses are printed out every 500 iterations, allowing you to monitor the model's progress during training. 

Whenever the validation loss reaches a new low, the model's state is saved to `char_transformer_language_model.pt`. Training stops early if the validation loss has not improved for 3 evaluations in a row (`patience`), and the best saved model is restored before generating text.

## Text Generation

Once trained, the model can be used to generate new text. The model takes a sequence of characters (the context) and generates the next character. This process is repeated to generate a sequence of new characters.

## Usage

Run the script from the repository root to train the model and generate new text:

```bash
python char_transformer_language_model.py
```

You can also run the provided Jupyter notebook, which contains the same code split into cells. The hyperparameters can be adjusted to fine-tune the model's performance and the amount of text generated.

## Future Work

Improvements to the model could include adjusting the hyperparameters, adding more transformer blocks, training on a larger dataset, or training for more iterations. 

## License

This project is open source and available under the [MIT License](LICENSE).

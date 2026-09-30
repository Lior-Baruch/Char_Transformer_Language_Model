# Character-Level Transformer Language Model

[![tests](https://github.com/Lior-Baruch/Char_Transformer_Language_Model/actions/workflows/tests.yml/badge.svg)](https://github.com/Lior-Baruch/Char_Transformer_Language_Model/actions/workflows/tests.yml)

A small, readable PyTorch library for experimenting with the whole LLM training pipeline, one character at a time:

```
                 pretrain                 SFT                      DPO
data/input.txt ───────────▶ base model ─────────▶ instruct model ─────────▶ preference-tuned model
(Shakespeare)             (writes Shakespeare)   (follows instructions) │
                                                                        │  GRPO
                                                                        └─────────▶ RL-tuned model
```

1. **Pretraining.** A GPT-style transformer learns to predict the next character of Shakespeare's plays.
2. **Supervised fine-tuning (SFT).** The base model learns a chat format and a set of instruction-following tasks, such as "Reverse the word: love" → "evol".
3. **Preference tuning (DPO).** The instruct model is trained on pairs of a correct answer and one of its own wrong answers.
4. **Reinforcement learning (GRPO).** The instruct model samples several answers per prompt and is rewarded for the correct ones.

Every stage runs on a laptop CPU. The example models in `checkpoints/example/` were trained that way in about 70 minutes in total, and each stage measurably improves on the one before (see [Results](#results-of-the-example-models)).

## Install

The only dependency is [PyTorch](https://pytorch.org/).

```bash
pip install -e .          # installs the `charlm` package and command
# or, without installing: run `python -m charlm ...` from the repository root
```

## Quick start

Chat with the example models, or compare them on held-out prompts:

```bash
python -m charlm chat --model checkpoints/example/grpo.pt "Reverse the word: prince"
python -m charlm chat --model checkpoints/example/sft.pt          # interactive
python -m charlm generate --model checkpoints/example/base.pt --prompt "ROMEO:" --max-new-tokens 300
python -m charlm eval --model checkpoints/example/{sft,dpo,grpo}.pt --show 2
```

Train the whole pipeline yourself (each stage reads the previous stage's checkpoint):

```bash
python -m charlm pretrain --config configs/example/pretrain.json   # ~35 min on 4 CPU cores
python -m charlm sft      --config configs/example/sft.json
python -m charlm dpo      --config configs/example/dpo.json
python -m charlm grpo     --config configs/example/grpo.json
```

Any config option can be overridden from the command line, which makes quick experiments easy:

```bash
python -m charlm grpo --config configs/example/grpo.json --set kl_coef=0 group_size=16 out_path=runs/grpo_nokl.pt
python -m charlm pretrain --config configs/example/pretrain.json --set model.n_layer=6 --print-config
```

Each run writes a checkpoint (`out_path`) and its metrics as JSON lines next to it (`*.metrics.jsonl`), ready for plotting and comparing runs. Every stage also saves its full training state at each evaluation (`*.state.pt`, deleted when the run finishes). If a run is interrupted, run the same command with `--set resume=true`; it continues exactly where it left off. `configs/pretrain_gpu.json` is the original 10.8M-parameter model, for use on a GPU.

## Results of the example models

All four models have 1.84M parameters (192-dim embeddings, 4 layers, 4 heads, 128-character context). They were trained on 4 CPU cores in about 70 minutes: pretraining 33 min, SFT 15, DPO 10, GRPO 11. Accuracy on 500 held-out prompts per task:

```bash
python -m charlm eval --model checkpoints/example/{base,sft,dpo,grpo}.pt --n-per-task 500
```

| model | reverse | uppercase | spell | length | add | overall |
|---|---|---|---|---|---|---|
| `base.pt` | 0% | 0% | 0% | 0% | 0% | 0% |
| `sft.pt` | 99.0% | 100% | 98.4% | 100% | 13.2% | 82.1% |
| `dpo.pt` | 95.8% | 97.8% | 97.6% | 100% | 20.2% | 82.3% |
| `grpo.pt` | 98.8% | 100% | 98.2% | 100% | **24.2%** | **84.2%** |

**Base model.** Val loss 1.477, the same as the original 10.8M-parameter notebook model (1.478) with a sixth of the parameters. It writes Shakespeare and ignores instructions:

```
ROMEO:
Still to us.

First Musician:
Ay, as I had not seen to-morrow?

ANGELO:
Beseech you, be not a love.
```

**SFT.** SFT teaches the chat format and solves the word tasks almost perfectly; "Reverse the word: shakespeare" → "eraepsekahs". It also keeps the Shakespeare: "Say a line as KING RICHARD III." → "Then I say, and so shall I stay."

Addition is the hard task. The model gets the size of the answer right (first digit 92%, number of digits 98%), but its last digit is close to a random guess, so only 13% of its answers are exact. That leaves room for the next two stages.

**DPO.** DPO raises addition from 13% to 20%, at a small cost on reverse and uppercase. This takes `nll_coef=1`. Plain DPO (`nll_coef=0`) made the model worse: overall 82% → 60%, addition 13% → 2%. The chosen and rejected answers differ by a single digit ("136" vs "132"), and DPO then pushes down the probability of both. Adding the next-token loss on the chosen answer, as in [RPO](https://arxiv.org/abs/2404.19733), prevents that. You can see it in `dpo.metrics.jsonl`: `train_chosen_reward` rises while `train_rejected_reward` falls.

**GRPO.** GRPO raises addition from 13% to 24% without hurting the other tasks. Over training, the reward on sampled replies doubles (8% → 17%). Three choices mattered:
- **It trains on addition only** (`"tasks": ["add"]`). The word tasks are already solved, so every reply in their groups gets the same reward and carries no learning signal. Trained on all tasks, only ~10% of groups had any signal, and addition didn't move (13.0%).
- **Groups of 16 replies.** With this group size, ~70% of addition groups contain both right and wrong answers.
- **A gentle update.** A higher learning rate (3e-4) with a weaker KL penalty (0.01) was unstable: the KL to the reference jumped to ~0.5 and the reward fell.

## Using it as a library

```python
from charlm import load_checkpoint, chat, TaskSuite, evaluate_tasks, load_config, SFTConfig, sft

model, tokenizer, meta = load_checkpoint("checkpoints/example/sft.pt")
print(chat(model, tokenizer, "Spell out: crown"))            # c-r-o-w-n

suite = TaskSuite(open("data/input.txt").read())
print(evaluate_tasks(model, tokenizer, suite.eval_set(100)))  # {'acc': ..., 'acc/reverse': ..., ...}

cfg = load_config(SFTConfig, "configs/example/sft.json", ["max_iters=500", "out_path=runs/sft_short.pt"])
model, tokenizer = sft(cfg)
```

The building blocks are exposed too: `CharTransformerLanguageModel` and `ModelConfig`, `CharTokenizer`, `sample_replies` (batched sampling), `encode_chat_example` (loss masking), `dpo.dpo_loss`, `grpo.grpo_loss`, `grpo.group_advantages`.

## How it works

### Tokenizer and chat template

Each character is one token. The vocabulary is every printable ASCII character and newline, so digits and symbols that never appear in Shakespeare can still be used later. It also has four special tokens. A conversation turn looks like this:

```
<|user|>Reverse the word: love<|assistant|>evol<|end|>
```

### Model

A decoder-only transformer (`charlm/model.py`):
- Token and position embeddings.
- Pre-norm transformer blocks, each with causal multi-head self-attention and a feed-forward network (ReLU, 4x wide).
- A final LayerNorm and a linear layer that outputs the next-token scores.

Attention is scaled by 1/sqrt(head_size).

### 1. Pretraining (`charlm/pretrain.py`)

- Random 128-character windows of the text are used as training inputs, and the target at every position is the next character.
- The last 10% of the text is held out as validation data.
- The learning rate follows a linear warmup, then a cosine decay.
- The checkpoint with the lowest validation loss is kept, and training stops early once the validation loss stops improving.

### 2. Supervised fine-tuning (`charlm/sft.py`)

- The model is trained on (prompt, response) pairs in the chat template.
- The loss is computed only on the response and `<|end|>` tokens. The prompt positions get target `-100`, so the model learns to answer rather than to imitate the user.
- The data is synthetic by default (see Tasks below). You can pass your own JSONL file of `{"prompt": ..., "response": ...}` rows with `data_path`.

### Tasks (`charlm/tasks.py`)

The instruction data is generated from the corpus. Five tasks have one correct answer, so a reply can be checked automatically:

| task | example prompt | answer |
|---|---|---|
| reverse | `Reverse the word: love` | `evol` |
| uppercase | `Write in capital letters: love` | `LOVE` |
| spell | `Spell out: love` | `l-o-v-e` |
| length | `How many letters are in "love"?` | `4` |
| add | `What is 38 + 45?` | `83` |
| speak | `Say a line as ROMEO.` | a line ROMEO speaks in the play (SFT only, not checkable) |

Because the answers can be checked, the same tasks give labeled data for SFT, correct/wrong pairs for DPO and a reward for GRPO.

Words are 3 to 12 letters long, and numbers go up to 99. 20% of the words, number pairs and speeches are held out. All accuracy numbers are measured on those held-out prompts, so they show whether the model learned the task rather than memorized the training examples.

### 3a. DPO (`charlm/dpo.py`)

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) trains on (prompt, chosen, rejected) triples. The loss is:

```
loss = -log sigmoid(beta * ((log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))))
```

- It raises the probability of the chosen reply and lowers the rejected one, relative to a frozen reference model (the SFT model).
- `beta` limits how far the model can move away from the reference.
- By default the pairs come from the model's own mistakes. The SFT model samples several replies per training prompt; for each prompt it got wrong at least once, the correct answer becomes "chosen" and a wrong reply becomes "rejected".
- The pairs are saved to `*.pairs.jsonl`. You can also supply your own pairs with `data_path`.
- `nll_coef` adds the ordinary next-token loss on the chosen reply. When chosen and rejected replies are nearly identical, plain DPO can lower the probability of both; this term keeps the chosen reply likely (see Results).

### 3b. GRPO (`charlm/grpo.py`)

[Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) is reinforcement learning with a verifiable reward and no value network. Each step works like this:

1. Sample `batch_size` training prompts from `tasks` and `group_size` replies to each, at temperature 1.
2. Reward each reply: 1 if it is correct, 0 if not.
3. Compute each reply's advantage relative to its own group: `(reward - group mean) / group std`. If a group's replies are all right or all wrong, they carry no signal.
4. Update the model with the PPO clipped objective on the reply tokens, plus a KL penalty (k3 estimator) that keeps it close to the reference model:

```
loss = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + kl_coef * KL(π || π_ref)
```

## Experiment ideas

- **Base model.** Scale the model or the context (`model.*`) and watch how the val loss and samples change. Or pretrain on your own text with `data_path`.
- **SFT.** Train on less data (`n_train`), train on a subset of `tasks` and test on the others, or skip pretraining and see how much the base model helps.
- **DPO.** Sweep `beta` and `nll_coef`. Or build the pairs from a different model than the one being trained (`data_path`).
- **GRPO.** Sweep `kl_coef` (try 0), `group_size` and `temperature`, or set `updates_per_batch` > 1 so the clipping matters. Or give addition partial credit per correct digit, or add a task in `tasks.py` with its own reward.
- **Chain the stages.** Run GRPO starting from the DPO model (`init_from`), or run DPO on pairs sampled from the GRPO model.

## Project layout

```
charlm/
  tokenizer.py    character tokenizer with chat special tokens
  model.py        the transformer, sampling, per-token log-probabilities
  checkpoint.py   save/load a model together with its config and tokenizer
  config.py       dataclass configs from JSON files + key=value overrides
  training.py     shared training utilities (optimizer, LR schedule, metrics logging)
  chat.py         chat template, loss masking, batched reply sampling
  tasks.py        synthetic instruction tasks with held-out splits and a reward
  evaluation.py   task accuracy
  pretrain.py     stage 1
  sft.py          stage 2
  dpo.py          stage 3a
  grpo.py         stage 3b
  cli.py          `python -m charlm ...`
configs/          example configs for each stage
.github/workflows/tests.yml   CI: lint and tests on pull requests and pushes to master
checkpoints/example/   trained example models, their metrics and the DPO pairs
data/input.txt    Tiny Shakespeare (1.1M characters)
tests/            pytest suite, including a tiny end-to-end run of the pipeline
char_transformer_language_model.ipynb   the original self-contained notebook walkthrough
```

Run the tests with `pip install pytest && pytest tests`. GitHub Actions also runs them, together with the `pyflakes` linter, on every pull request and every push to `master` (`.github/workflows/tests.yml`).

## The notebook

`char_transformer_language_model.ipynb` builds and trains the pretraining model step by step in a single file, without the library. It is a good place to start reading.

## License

This project is open source and available under the [MIT License](LICENSE).

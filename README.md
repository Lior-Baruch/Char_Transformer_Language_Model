# Character-Level Transformer Language Model

[![tests](https://github.com/Lior-Baruch/Char_Transformer_Language_Model/actions/workflows/tests.yml/badge.svg)](https://github.com/Lior-Baruch/Char_Transformer_Language_Model/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lior-Baruch/Char_Transformer_Language_Model/blob/master/notebooks/colab_pipeline.ipynb)

A small, readable PyTorch library for experimenting with the whole LLM training pipeline, one character at a time:

1. **Pretraining.** A GPT-style transformer learns to predict the next character of Shakespeare's plays.
2. **Supervised fine-tuning (SFT).** The base model learns a chat format and a set of instruction-following tasks, such as "Reverse the word: love" → "evol".
3. **Preference tuning (DPO).** The instruct model is trained on pairs of a correct answer and one of its own wrong answers.
4. **Reinforcement learning (GRPO).** The instruct model samples several answers per prompt and is rewarded for the correct ones.
5. **Reasoning.** With `reasoning: true`, SFT teaches the model to work out math problems step by step, between `<|think|>` and `<|/think|>`, before it answers.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/example_models_dark.png">
  <img alt="Diagram of the example models. data/input.txt (Tiny Shakespeare, 1.1M characters) is pretrained into base.pt (writes Shakespeare, val loss 1.48, 0% on the tasks). SFT turns it into sft.pt (82% on 5 tasks, addition 13%), which DPO turns into dpo.pt (82%, addition 20%) and GRPO into grpo.pt (84%, addition 24%). SFT with reasoning turns base.pt into sft_reasoning.pt (99-100% on addition, subtraction, multiplication and division; word problems 22%), which GRPO turns into grpo_reasoning.pt (word problems 34%)." src="docs/figures/example_models.png">
</picture>

- **Everything runs on a laptop CPU.** The four main models in `checkpoints/example/` (`base`, `sft`, `dpo`, `grpo`) took about 70 minutes to train, and the two reasoning models about 75 more. Each stage after SFT improves addition: 13% after SFT, 20% after DPO, 24% after GRPO (see [Results](#results-of-the-example-models)).
- **Step-by-step reasoning makes arithmetic work.** The same 1.8M-parameter base model, fine-tuned two ways, gets 4-31% of held-out additions, subtractions, multiplications and divisions right (problems it never saw in training) when trained to answer directly, and 99-100% when trained to write out the steps (see [Reasoning](#reasoning-step-by-step)).
- **A bigger model on a GPU.** [A Colab notebook](#a-bigger-model-on-a-gpu-colab) pretrains a 59M-parameter model on a billion characters of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and fine-tunes it to reason.

**Contents:** [Install](#install) · [Quick start](#quick-start) · [Colab](#a-bigger-model-on-a-gpu-colab) · [Your own data](#more-training-data) · [Results](#results-of-the-example-models) · [Reasoning](#reasoning-step-by-step) · [Library](#using-it-as-a-library) · [How it works](#how-it-works) · [Experiment ideas](#experiment-ideas) · [Project layout](#project-layout) · [Original notebook](#the-original-notebook)

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
python -m charlm eval --model checkpoints/example/sft.pt checkpoints/example/dpo.pt checkpoints/example/grpo.pt --show 2
```

The reasoning model shows its work before answering:

```bash
python -m charlm chat --model checkpoints/example/sft_reasoning.pt "What is 386 * 7?"
# [thinking: 6*7=42+0=42 A=2, 8*7=56+4=60 A=02, 3*7=21+6=27 A=2702 => 2702] 2702
python -m charlm chat --model checkpoints/example/sft_reasoning.pt "Adam has 12 apples to divide equally among 3 friends. How many each?"
# [thinking: 12/3: 01/3=0 r1 A=0, 12/3=4 r0 A=04 => 4] 4
```

Train the whole pipeline yourself. SFT starts from `base.pt`, and DPO and GRPO each start from `sft.pt` (two alternative ways to improve it):

```bash
python -m charlm pretrain --config configs/example/pretrain.json   # ~35 min on 4 CPU cores
python -m charlm sft      --config configs/example/sft.json
python -m charlm dpo      --config configs/example/dpo.json
python -m charlm grpo     --config configs/example/grpo.json
```

These configs write to `checkpoints/example/`, so they replace the models that come with the repository (`git checkout checkpoints/example` restores them). To keep those, give each stage its own output file and point the next stages at it:

```bash
python -m charlm pretrain --config configs/example/pretrain.json --set out_path=runs/base.pt
python -m charlm sft      --config configs/example/sft.json  --set init_from=runs/base.pt out_path=runs/sft.pt
python -m charlm dpo      --config configs/example/dpo.json  --set init_from=runs/sft.pt out_path=runs/dpo.pt
python -m charlm grpo     --config configs/example/grpo.json --set init_from=runs/sft.pt out_path=runs/grpo.pt
```

Any config option can be overridden from the command line, which makes quick experiments easy:

```bash
python -m charlm grpo --config configs/example/grpo.json --set kl_coef=0 group_size=16 out_path=runs/grpo_nokl.pt
python -m charlm pretrain --config configs/example/pretrain.json --set model.n_layer=6 --print-config
```

Each run writes a checkpoint (`out_path`) and its metrics as JSON lines next to it (`*.metrics.jsonl`), ready for plotting and comparing runs. If a run is interrupted, run the same command with `--set resume=true`.

<details>
<summary>Resuming, in detail</summary>

Every stage saves its full training state at each evaluation (`*.state.pt`, deleted when the run finishes; `state_every=N` saves it every N evaluations instead, for big models). On a CPU it continues exactly where it left off, bit for bit. On a GPU the random-number state is restored too, but some GPU operations aren't deterministic, so the numbers can differ slightly. With `resume=true`, a stage that already finished with the same settings is skipped, so a whole pipeline can simply be rerun after a crash. If the settings changed, resuming stops with an error instead of mixing two different runs (or keeping a finished model trained with other settings): delete the old checkpoint, or set `resume=false`, to train again.

</details>

Training uses the GPU automatically when there is one (`device` defaults to `auto`). `precision` sets the number format: `fp32` (the default), `bf16`, `fp16`, or `auto`, which picks bf16 on GPUs that support it natively (A100, L4, RTX 30xx and newer), fp16 with loss scaling on older ones (T4, V100) and fp32 on a CPU. 16-bit training is several times faster on a GPU. `configs/pretrain_gpu.json` is the original 10.8M-parameter model, for use on a GPU.

## A bigger model on a GPU (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lior-Baruch/Char_Transformer_Language_Model/blob/master/notebooks/colab_pipeline.ipynb)

`notebooks/colab_pipeline.ipynb` runs the pipeline on a Colab GPU with the configs in `configs/colab/`:

| stage | what it does | rough time on an A100 |
|---|---|---|
| data | download TinyStories and keep the first billion characters | ~5 min |
| pretrain | 59M parameters (12 layers, 640-dim, 10 heads, 512-character context), 60,000 steps of 64 x 512 characters | ~2 h |
| SFT | all nine checkable tasks, with reasoning | ~20 min |
| GRPO | the five math tasks, 32 prompts x 16 replies per step | ~30 min |

A last, optional cell runs DPO instead of GRPO (`configs/colab/dpo.json`), on pairs of the correct reasoning and one of the model's own wrong replies.

The times are estimates: this repository's own runs are on a CPU. An L4 is roughly three times slower, so the notebook has a `MAX_ITERS` setting to shorten pretraining. Training runs in bf16 (`"precision": "auto"`). Data and checkpoints are kept on Google Drive. If Colab disconnects, run all the cells again: finished stages are skipped and the interrupted one resumes from its last saved state.

## More training data

`prepare-data` downloads a corpus and cleans it to the tokenizer's characters (printable ASCII and newline: curly quotes become straight ones, accents are dropped, and so on):

```bash
python -m charlm prepare-data tinystories                      # ~2.7M short stories, ~2.2 GB -> data/tinystories.txt
python -m charlm prepare-data tinystories --max-chars 50000000  # just the first 50M characters
python -m charlm prepare-data shakespeare                      # the complete works, 5x data/input.txt -> data/shakespeare.txt
python -m charlm prepare-data files --files my_texts/*.txt --out data/mine.txt   # your own text files
python -m charlm pretrain --config configs/example/pretrain.json --set data_path=data/shakespeare.txt
```

The data is streamed, so it is never held in memory whole. A `.json` file next to the output records what was prepared, so running the same command again skips the download. The SFT tasks still take their words from `corpus_path` (`data/input.txt` by default), whatever the base model was pretrained on.

## Results of the example models

The four main models have 1.84M parameters (192-dim embeddings, 4 layers, 4 heads, 128-character context). They were trained on 4 CPU cores in about 70 minutes: pretraining 33 min, SFT 15, DPO 10, GRPO 11. Accuracy on 500 held-out prompts per task:

```bash
python -m charlm eval --model checkpoints/example/base.pt checkpoints/example/sft.pt checkpoints/example/dpo.pt checkpoints/example/grpo.pt --n-per-task 500
```

| model | reverse | uppercase | spell | length | add | overall |
|---|---|---|---|---|---|---|
| `base.pt` | 0% | 0% | 0% | 0% | 0% | 0% |
| `sft.pt` | 99.0% | 100% | 98.4% | 100% | 13.2% | 82.1% |
| `dpo.pt` | 95.8% | 97.8% | 97.6% | 100% | 20.2% | 82.3% |
| `grpo.pt` | 98.8% | 100% | 98.2% | 100% | **24.2%** | **84.2%** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/final_comparison_dark.png">
  <img alt="Grouped bar chart of held-out accuracy per task for the SFT, DPO and GRPO models. All three are near 100% on reverse, uppercase, spell and length. On addition SFT scores 13%, DPO 20% and GRPO 24%." src="docs/figures/final_comparison.png">
</picture>

The accuracy curves below (SFT, DPO and GRPO) are measured during training on 100 held-out prompts per task (50 for SFT), so they are noisier than the 500-prompt table. The figures are drawn by `docs/make_figures.py`, `docs/make_reasoning_figures.py` and `docs/make_diagrams.py`, and the example tables printed by `docs/make_examples.py`, from the files in `checkpoints/example/` (`pip install matplotlib`, then e.g. `python docs/make_figures.py`).

### Pretraining

The base model reaches val loss 1.477, the same as the original 10.8M-parameter notebook model (1.478) with a sixth of the parameters. Validation loss stopped improving after step 3,000 while training loss kept falling (the model starts to memorize the training text), so early stopping ended the run at step 3,750 and kept the best checkpoint, from step 3,000.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/pretraining_loss_dark.png">
  <img alt="Line chart of training and validation loss over 3,750 pretraining steps. Both fall quickly at first; validation loss flattens around 1.48, reaching its best, 1.477, at step 3,000, while training loss keeps falling to 1.16." src="docs/figures/pretraining_loss.png">
</picture>

It writes Shakespeare and ignores instructions. Given "ROMEO:" it continues with a scene:

```
ROMEO:
Still to us.

First Musician:
Ay, as I had not seen to-morrow?

ANGELO:
Beseech you, be not a love.
```

Asked "Reverse the word: toy", it just keeps writing a play (" the common of the strength…"). That's why it scores 0% on every task.

### SFT

SFT teaches the chat format and the word tasks: they reach 84-100% accuracy by step 1,000 and 98-100% by step 2,000. "Reverse the word: shakespeare" → "eraepsekahs". Its replies to "Say a line as …" sound Shakespearean, if not very meaningful (examples below).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/sft_accuracy_dark.png">
  <img alt="Five small line charts of held-out accuracy during SFT, one per task. Reverse, uppercase, spell and length reach 84-100% by step 1,000 and 98-100% by step 2,000; addition stays between 0% and 16%." src="docs/figures/sft_accuracy.png">
</picture>

Addition is the hard task. The model gets the size of the answer right (first digit 92%, number of digits 98%), but its last digit is close to a random guess, so only 13% of its answers are exact. That leaves room for the next two stages.

### DPO

DPO raises addition from 13% to 20%, at a small cost on reverse and uppercase. This takes `nll_coef=1`. Plain DPO (`nll_coef=0`) made the model worse: overall 82% → 60%, addition 13% → 2%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/dpo_comparison_dark.png">
  <img alt="Two line charts over 500 DPO steps. Overall accuracy: DPO with NLL stays at 82-84% while plain DPO falls to 60%. Addition accuracy: DPO with NLL rises from 9% to 25% while plain DPO falls to 2%." src="docs/figures/dpo_comparison.png">
</picture>

The reason is in the pairs. They come from the SFT model's own mistakes, and for addition the correct and wrong answers differ by a digit or two:

| prompt | chosen (correct answer) | rejected (the SFT model's own sampled answer) |
|---|---|---|
| What is 98 + 38? | `136` | `132` |
| What is 15 + 60? | `75` | `74` |
| What is 31 + 96? | `127` | `120` |
| Spell out: universal | `u-n-i-v-e-r-s-a-l` | `u-n-i-v-e-r-s-Al-l` |
| Spell out: beam | `b-e-a-m` | `b-e-n-m` |
| How many letters are in "courtship"? | `9` | `2` |

Plain DPO then pushes down the probability of both the chosen and the rejected answer. Adding the next-token loss on the chosen answer, as in [RPO](https://arxiv.org/abs/2404.19733), prevents that. You can see it in `dpo.metrics.jsonl`: `train_chosen_reward` rises while `train_rejected_reward` falls. All 4,000 pairs are in `checkpoints/example/dpo.pairs.jsonl`.

### GRPO

GRPO raises addition from 13% to 24% without hurting the other tasks. Over training, the reward on sampled replies doubles (8% → 17%).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/grpo_comparison_dark.png">
  <img alt="Two line charts comparing three GRPO runs. Addition accuracy: the final run (addition only, lr 1e-4, KL 0.04) rises from 9% to 23% over 600 steps; training on all tasks stays near 11%; the lr 3e-4 run drops to 4% and was stopped. KL from the SFT model: the lr 3e-4 run jumps to 0.82 within 150 steps, the final run stays between 0.06 and 0.3, the all-tasks run stays near 0.01." src="docs/figures/grpo_comparison.png">
</picture>

Three choices mattered:
- **It trains on addition only** (`"tasks": ["add"]`). The word tasks are already solved, so every reply in their groups gets the same reward, and the only thing that acts on them is the KL penalty. Trained on all tasks, only ~10% of groups had any reward signal, and addition didn't move (13.0%).
- **Groups of 16 replies.** With this group size, ~70% of addition groups contain both right and wrong answers.
- **A gentle update.** A higher learning rate (3e-4) with a weaker KL penalty (0.01) was unstable: the KL to the reference jumped past 0.8 and the reward fell.

The logs of the two runs that didn't work are in `checkpoints/example/experiments/`.

<details>
<summary>What the GRPO training log looks like</summary>

```
device cpu | 16 prompts x 16 replies per step | 500 eval prompts
step 0 | acc 0.816 | acc/reverse 1 | acc/uppercase 1 | acc/spell 0.99 | acc/length 1 | acc/add 0.09 | 3s
step 25 | reward 0.07969 | kl 0.0637 | loss 0.002548 | reply_len 3.484 | groups_with_signal 0.63 | acc 0.824 | acc/reverse 1 | acc/uppercase 1 | acc/spell 0.99 | acc/length 1 | acc/add 0.13 | 32s
step 300 | reward 0.1064 | kl 0.1257 | loss 0.005026 | reply_len 3.505 | groups_with_signal 0.675 | acc 0.822 | acc/reverse 0.99 | acc/uppercase 0.99 | acc/spell 0.98 | acc/length 1 | acc/add 0.15 | 315s
step 600 | reward 0.1698 | kl 0.2222 | loss 0.008886 | reply_len 3.483 | groups_with_signal 0.7475 | acc 0.836 | acc/reverse 0.99 | acc/uppercase 0.99 | acc/spell 0.97 | acc/length 1 | acc/add 0.23 | 628s
saved model to checkpoints/example/grpo.pt
```

- `reward`: the share of sampled replies that were correct in this step's groups.
- `kl`: how far the model has moved from the SFT model, per token.
- `groups_with_signal`: the share of groups with both right and wrong replies. Only those have a reward signal; the KL penalty acts on every group, but it only pulls the model back toward the SFT model.
- `acc/...`: greedy accuracy on held-out prompts, measured every 25 steps.

</details>

### Example replies

These are the first prompts of each task in the 500-per-task evaluation set above, not hand-picked, answered greedily by each model:

| prompt | expected | SFT | DPO | GRPO |
|---|---|---|---|---|
| Reverse the word: toy | `yot` | `yot` ✓ | `yot` ✓ | `yot` ✓ |
| Reverse the word: faster | `retsaf` | `retsaf` ✓ | `retsaf` ✓ | `retsaf` ✓ |
| Write in capital letters: oppression | `OPPRESSION` | `OPPRESSION` ✓ | `OPPRESSION` ✓ | `OPPRESSION` ✓ |
| Spell out: pipes | `p-i-p-e-s` | `p-i-p-e-s` ✓ | `p-i-p-e-s` ✓ | `p-i-p-e-s` ✓ |
| Spell out: meteors | `m-e-t-e-o-r-s` | `m-e-t-e-o-r-s` ✓ | `m-e-t-e-o-r-s` ✓ | `m-e-t-e-o-r-s` ✓ |
| How many letters are in "conqueror"? | `9` | `9` ✓ | `9` ✓ | `9` ✓ |
| What is 65 + 11? | `76` | `72` ✗ | `77` ✗ | `77` ✗ |
| What is 32 + 95? | `127` | `122` ✗ | `127` ✓ | `122` ✗ |
| What is 27 + 41? | `68` | `66` ✗ | `67` ✗ | `61` ✗ |
| What is 33 + 96? | `129` | `122` ✗ | `126` ✗ | `122` ✗ |
| What is 53 + 19? | `72` | `72` ✓ | `70` ✗ | `72` ✓ |

The word tasks are easy for all three. On addition, every wrong answer here has the right leading digits and a wrong last digit, as described under SFT. Each model gets one of these five right; five prompts are too few to show the differences in the table above.

The "Say a line as …" task has no single right answer, so DPO and GRPO don't train on it. SFT and GRPO replies:

| prompt | SFT | GRPO |
|---|---|---|
| Say a line as ROMEO. | I would I have something the countest of him. | I would I have so. |
| Say a line as JULIET. | I would not her had something the city of him. | I would you have some to have some offer'd |
| Say a line as KING RICHARD III. | What is the country the countest? | Then I say, and so shall I stay. |
| Say a line as Nurse. | I would I have so so. | I would I have so. |

GRPO gives the same line for two different speakers.

## Reasoning step by step

Does writing out the steps help a 1.8M-parameter model do arithmetic? To find out, two SFT runs start from the same `base.pt`, with the same 100,000 examples of nine tasks (`reverse`, `uppercase`, `spell`, `length` and the math tasks `add`, `sub`, `mul`, `div`, `word`), the same 5,000 steps and the same batch size. The only difference is the replies to the math tasks: the plain model answers directly, the reasoning model first writes the [scratchpad](#reasoning-tokens). Both then get 300 steps of GRPO on the five math tasks (16 prompts x 8 replies). The plain model is its own run on the nine tasks, not the example `sft.pt` (which trained on five tasks and scores 13% on addition). Accuracy on 500 held-out prompts per task:

```bash
python -m charlm sft  --config configs/reasoning/sft_reasoning.json   # ~65 min on 2 CPU cores
python -m charlm grpo --config configs/reasoning/grpo_reasoning.json  # ~13 min
python -m charlm sft  --config configs/reasoning/sft_plain.json       # ~35 min; the plain models are not in the
python -m charlm grpo --config configs/reasoning/grpo_plain.json      # ~5 min   repository, only their metrics
python -m charlm eval --model checkpoints/example/sft_reasoning.pt --n-per-task 500 --max-new-tokens 100 --show 9 \
    --tasks reverse uppercase spell length add sub mul div word
```

Like the example configs, the reasoning configs write to `checkpoints/example/` and replace the models there.

| model | reverse | uppercase | spell | length | add | sub | mul | div | word | overall |
|---|---|---|---|---|---|---|---|---|---|---|
| plain SFT | 99.2% | 100% | 100% | 100% | 8.0% | 4.4% | 30.6% | 20.6% | 4.4% | 51.9% |
| plain SFT + GRPO | 98.8% | 100% | 99.8% | 100% | 10.4% | 5.6% | 31.0% | 24.4% | 3.4% | 52.6% |
| SFT + reasoning (`sft_reasoning.pt`) | 99.2% | 99.4% | 99.4% | 100% | **99.6%** | **100%** | **100%** | 99.0% | 21.6% | 90.9% |
| SFT + reasoning + GRPO (`grpo_reasoning.pt`) | 99.0% | 99.4% | 99.4% | 100% | **99.6%** | **100%** | **100%** | **99.4%** | **33.8%** | **92.3%** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_final_dark.png">
  <img alt="Grouped bar chart of held-out accuracy of four models on add, sub, mul, div, word and all nine tasks. The plain SFT model scores 4% to 31% on the arithmetic tasks, and GRPO barely changes that. Both reasoning models score 99% to 100%. On word problems the plain models score under 5%, the reasoning model 22% and 34% after GRPO. Overall: 52%, 53%, 91% and 92%." src="docs/figures/reasoning_final.png">
</picture>

**The scratchpad solves the arithmetic.** Without it, SFT gets 8% of held-out additions right, 4% of subtractions, 31% of multiplications and 21% of divisions. With it, 99-100% on all four, with numbers up to 999 (99 for addition, and up to 8991 / 9 for division). The model, the prompts and the number of steps are the same. The model also follows the method exactly: on add, sub, mul and div, 99-100% of its scratchpads are the taught trace, character for character.

Where does the plain model go wrong? It gets the size of the answer right: 91-100% of its answers have the right number of digits, and when an answer has a hundreds digit (a thousands digit for multiplication), that digit is right 89-97% of the time. The lower digits are much worse: an addition's ones digit is right only 9% of the time, about a random guess, even though it depends on nothing but the two ones digits. A subtraction's tens and ones digits are right 20% and 13% of the time. Multiplication is the exception at the end: its ones digit is right almost always, because the last digit of `386 * 7` only depends on `6 * 7`, a times-table fact. Its middle digits are right only half the time. With a scratchpad, each digit gets its own step, computed from the two digits it combines and the carry written just before it: `8*7=56+4=60`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_digits_dark.png">
  <img alt="Four small grouped bar charts, one per arithmetic task, showing the share of held-out answers with each digit right, by place value. The reasoning model gets every digit right 99-100% of the time. The plain model gets the highest place value right 89-97% of the time, but the ones digit of an addition only 9%, the tens and ones of a subtraction 20% and 13%, the middle digits of a multiplication 54% and 50% (its ones digit 100%), and the tens and ones of a division 33% and 66%." src="docs/figures/reasoning_digits.png">
</picture>

During training, the reasoning model reaches 100% on multiplication first (step 2,000) and on division last (step 4,250), while the plain model stays at or below 28% on every math task.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_sft_dark.png">
  <img alt="Five small line charts of held-out accuracy during SFT on add, sub, mul, div and word, for the plain and the reasoning model. The reasoning model reaches 100% on mul by step 2,000, on add and sub by step 3,250 and on div by step 4,250. The plain model stays at or below 28%. On word problems the reasoning model stays between 16% and 28% after step 1,000, and the plain one at or below 6%." src="docs/figures/reasoning_sft.png">
</picture>

**The price is length.** A reasoning reply averages 39-73 tokens per math task, against 3-5 for a direct answer, about 15 times longer. Training on the longer replies took almost twice as long on the same hardware (63 minutes instead of 36), and every answer takes that many more steps to generate.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_cost_dark.png">
  <img alt="Horizontal bar chart of average reply length per math task. Plain SFT replies are 3-5 tokens long and 4-31% right; reasoning replies are 39 tokens for add (100% right), 59 for sub (100%), 66 for mul (100%), 73 for div (99%) and 45 for word problems (22%)." src="docs/figures/reasoning_cost.png">
</picture>

**Word problems are about language, not arithmetic.** The reasoning model gets 22% of the held-out word problems. They are asked in a fourth phrasing it never saw. The same problems asked in one of its three training phrasings get 98%:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_word_dark.png">
  <img alt="Dot plot of word-problem accuracy per operation. In the phrasings it trained on, the reasoning model gets 100% of addition, subtraction and multiplication stories and 92% of division stories. In the new phrasing it gets 25% of addition (45% after GRPO), 2% of subtraction (1% after GRPO), 1% of multiplication (22% after GRPO) and 60% of division (71% after GRPO)." src="docs/figures/reasoning_word.png">
</picture>

<details>
<summary>The numbers</summary>

| model | phrasing | accuracy | first equation has the right value | + | - | * | / |
|---|---|---|---|---|---|---|---|
| SFT + reasoning | never trained on | 22% | 28% | 25% | 2% | 1% | 60% |
| SFT + reasoning | trained on | 98% | 100% | 100% | 100% | 100% | 92% |
| SFT + reasoning + GRPO | never trained on | 34% | 33% | 45% | 1% | 22% | 71% |
| SFT + reasoning + GRPO | trained on | 97% | 100% | 100% | 100% | 100% | 89% |

The equation column checks the equation the model writes first (like `12/3:`) by its result. A few replies write an equation with the wrong result but still reach the right answer, so accuracy can be slightly higher than that column.

</details>

With the new phrasing, the model mostly writes the wrong equation first. It reads "There are 7 marbles and 8 more arrive" as `87+8`, and "Adam puts 20 stickers into 5 equal groups" as `20+5`. Three phrasings per operation are not enough for a character-level model to learn what the words mean. More phrasings, or a bigger model pretrained on more English (like the [Colab model](#a-bigger-model-on-a-gpu-colab)), are the natural next experiments.

**GRPO has little left to teach, but it helps with the new phrasing.** The reasoning model already gets 99% of its sampled training replies right, so only ~5% of the groups contain both a right and a wrong reply, and only those carry a reward signal (the KL penalty acts on all of them, but it only pulls the model back toward the SFT model). The arithmetic stays at 99-100%. Still, held-out word problems rise from 22% to 34%, although GRPO trains only on the three training phrasings. The gains have different causes. Multiplication stories go from 1% to 22% because the model now reads them right: its first equation is right 22% of the time instead of 1%. Addition stories go from 25% to 45% although the model writes the right equation just as often (50% before, 49% after): it now finishes the sum it wrote more often. The plain model's GRPO run goes nowhere: its reward stays near 20%, and no task's held-out accuracy moves by more than 4 points. GRPO took 5 minutes for the plain model and 13 for the reasoning one.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning_grpo_dark.png">
  <img alt="Three line charts over 300 GRPO steps. Reward: the reasoning model gets 98-99% of its sampled replies right throughout, the plain model 16-22%. Groups with both right and wrong replies: 31-40% for the plain model, 3-8% for the reasoning model. Held-out word problems in the new phrasing: the reasoning model rises from 18% to 28% (50 prompts), the plain model stays at 0-4%." src="docs/figures/reasoning_grpo.png">
</picture>

Replies of the plain and the reasoning SFT model to the first held-out prompts of each math task (greedy):

| prompt | expected | plain SFT | SFT + reasoning |
|---|---|---|---|
| What is 65 + 11? | `76` | `78` ✗ | `<\|think\|>5+1+0=06 A=6, 6+1+0=07 A=076 => 76<\|/think\|>76` ✓ |
| What is 83 + 41? | `124` | `127` ✗ | `<\|think\|>3+1+0=04 A=4, 8+4+0=12 A=124 => 124<\|/think\|>124` ✓ |
| What is 324 - 172? | `152` | `177` ✗ | `<\|think\|>4-2-0=2 b0 A=2, 2-7-0=5 b1 A=52, 3-1-1=1 b0 A=152 => 152<\|/think\|>152` ✓ |
| What is 585 - 559? | `26` | `10` ✗ | `<\|think\|>5-9-0=6 b1 A=6, 8-5-1=2 b0 A=26, 5-5-0=0 b0 A=026 => 26<\|/think\|>26` ✓ |
| What is 370 * 2? | `740` | `740` ✓ | `<\|think\|>0*2=00+0=00 A=0, 7*2=14+0=14 A=40, 3*2=06+1=07 A=0740 => 740<\|/think\|>740` ✓ |
| What is 139 * 7? | `973` | `973` ✓ | `<\|think\|>9*7=63+0=63 A=3, 3*7=21+6=27 A=73, 1*7=07+2=09 A=0973 => 973<\|/think\|>973` ✓ |
| What is 468 / 6? | `78` | `81` ✗ | `<\|think\|>04/6=0 r4 A=0, 46/6=7 r4 A=07, 48/6=8 r0 A=078 => 78<\|/think\|>78` ✓ |
| What is 3520 / 8? | `440` | `480` ✗ | `<\|think\|>03/8=0 r3 A=0, 35/8=4 r3 A=04, 32/8=4 r0 A=044, 00/8=0 r0 A=0440 => 440<\|/think\|>440` ✓ |
| Omer puts 12 marbles into 4 equal groups. How many per group? | `3` | `5` ✗ | `<\|think\|>12/4: 01/4=0 r1 A=0, 12/4=3 r0 A=03 => 3<\|/think\|>3` ✓ |
| Lily packs 5 cards into each of 3 boxes. How many cards? | `15` | `25` ✗ | `<\|think\|>55/3: 05/3=1 r0 A=1, 05/3=1 r0 A=11 => 11<\|/think\|>11` ✗ |

The last one shows the word-problem failure: in the unfamiliar phrasing, the model reads "5 cards into each of 3 boxes" as a division, `55/3`.

The plain models are not in the repository, only their metrics (`checkpoints/example/experiments/`); `docs/make_reasoning_figures.py` makes the figures and tables from them.

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

Each character is one token. The vocabulary is every printable ASCII character and newline, so digits and symbols that never appear in Shakespeare can still be used later. It also has four special tokens (six with the [reasoning tokens](#reasoning-tokens)). A conversation turn looks like this:

```
<|user|>Reverse the word: love<|assistant|>evol<|end|>
```

### Model

A decoder-only transformer (`charlm/model.py`):
- Token and position embeddings.
- Pre-norm transformer blocks, each with causal multi-head self-attention and a feed-forward network (ReLU, 4x wide).
- A final LayerNorm and a linear layer that outputs the next-token scores.

Attention is scaled by 1/sqrt(head_size). When generating, a KV cache keeps each layer's keys and values, so each new token costs one position instead of the whole context.

### 1. Pretraining (`charlm/pretrain.py`)

- Random 128-character windows of the text are used as training inputs, and the target at every position is the next character.
- The last 10% of the text is held out as validation data.
- `data_path` can be a file, a directory of `.txt` files, a pattern like `"data/*.txt"`, or a list of these. With several files, the end of each file is held out. `max_val_chars` caps that part (the Colab config holds out 2M characters rather than 10% of a billion). ASCII text is stored as one byte per character, so a 1 GB corpus fits in 1 GB of memory.
- The learning rate follows a linear warmup, then a cosine decay.
- The checkpoint with the lowest validation loss is kept, and training stops early once the validation loss stops improving.

### 2. Supervised fine-tuning (`charlm/sft.py`)

- The model is trained on (prompt, response) pairs in the chat template.
- The loss is computed only on the response and `<|end|>` tokens. The prompt positions get target `-100`, so the model learns to answer rather than to imitate the user.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/sft_tokens_dark.png">
  <img alt="Diagram of one SFT example as a row of tokens. The prompt, <|user|>What is 47 + 85?<|assistant|>, is gray: no loss. The reply is colored: the reasoning <|think|>7+5+0=12 A=2, 4+8+1=13 A=132 => 132<|/think|> in orange and the answer 132<|end|> in blue. The model is trained to predict every reply token." src="docs/figures/sft_tokens.png">
</picture>

- The data is synthetic by default (see Tasks below). You can pass your own JSONL file of `{"prompt": ..., "response": ...}` rows with `data_path`.

### Tasks (`charlm/tasks.py`)

The instruction data is generated from the corpus. Every task except `speak` has one correct answer, so a reply can be checked automatically:

| task | example prompt | answer |
|---|---|---|
| reverse | `Reverse the word: love` | `evol` |
| uppercase | `Write in capital letters: love` | `LOVE` |
| spell | `Spell out: love` | `l-o-v-e` |
| length | `How many letters are in "love"?` | `4` |
| add | `What is 38 + 45?` | `83` |
| sub | `What is 704 - 358?` | `346` |
| mul | `What is 386 * 7?` | `2702` |
| div | `What is 912 / 8?` | `114` |
| word | `Adam has 12 apples to divide equally among 3 friends. How many each?` | `4` |
| speak | `Say a line as ROMEO.` | a line ROMEO speaks in the play (SFT only, not checkable) |

Because the answers can be checked, the same tasks give labeled data for SFT, correct/wrong pairs for DPO and a reward for GRPO.

The first five tasks are the defaults. `sub`, `mul`, `div` and `word` are used when a config lists them in `tasks`. Words are 3 to 12 letters long and `add` uses numbers up to 99. `sub` uses numbers up to 999 (never below zero), `mul` multiplies a number up to 999 by one digit, and `div` divides by one digit with no remainder. Word problems use one operation on small numbers, in four phrasings per operation.

20% of the words, number problems and speeches are held out, and so is one phrasing of each word problem. All accuracy numbers are measured on those held-out prompts, so they show whether the model learned the task rather than memorized the training examples (or, for word problems, the phrasing).

### Reasoning tokens

With `reasoning: true`, SFT adds two special tokens, `<|think|>` and `<|/think|>`, and the replies to the math tasks show their work before the answer. The scratchpads work digit by digit, like arithmetic on paper, so each step is small enough for a tiny model to learn:

```
What is 47 + 85?   <|think|>7+5+0=12 A=2, 4+8+1=13 A=132 => 132<|/think|>132
What is 82 - 47?   <|think|>2-7-0=5 b1 A=5, 8-4-1=3 b0 A=35 => 35<|/think|>35
What is 47 * 6?    <|think|>7*6=42+0=42 A=2, 4*6=24+4=28 A=282 => 282<|/think|>282
What is 84 / 6?    <|think|>08/6=1 r2 A=1, 24/6=4 r0 A=14 => 14<|/think|>14
Maya has 12 cookies to divide equally among 3 friends. How many each?
                   <|think|>12/3: 01/3=0 r1 A=0, 12/3=4 r0 A=04 => 4<|/think|>4
```

- Addition and multiplication go from the rightmost digit, writing the carry (`+1`). Subtraction writes the borrow (`b1`). Division goes from the left, writing the remainder (`r2`).
- `A=` is the answer so far. The trace ends with `=> answer`, which the model then copies after `<|/think|>`.
- Word problems first write the equation (`12/3:`), so the model has to work out which operation the story needs.
- Only the final answer is scored. The evaluation also reports `trace/<task>`, the share of replies whose reasoning matches the taught method exactly.

The tokens are appended after the existing ones, so a model without them (like `base.pt`) keeps every token id. The model's embedding grows by two rows (`model.resize_vocab`). DPO builds its chosen replies with the reasoning when the model has these tokens, and GRPO logs `closed_think`, the share of sampled replies that finish their reasoning within `max_new_tokens`. A reply that is cut off has no answer and gets reward 0.

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
3. Compute each reply's advantage relative to its own group: `(reward - group mean) / group std`. If a group's replies are all right or all wrong, their advantages are all 0: they carry no reward signal, and only the KL penalty below acts on them.
4. Update the model with the PPO clipped objective on the reply tokens, plus a KL penalty (k3 estimator) that keeps it close to the reference model:

```
loss = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + kl_coef * KL(π || π_ref)
```

## Experiment ideas

- **Base model.** Scale the model or the context (`model.*`) and watch how the val loss and samples change. Or pretrain on more text (`prepare-data`) or your own (`data_path`).
- **Reasoning.** Change the scratchpad format in `reasoning.py` (e.g. drop the running answer `A=`, or the carries) and see which parts the model needs. Or train on 2-digit numbers and test on 3-digit ones.
- **SFT.** Train on less data (`n_train`), train on a subset of `tasks` and test on the others, or skip pretraining and see how much the base model helps.
- **DPO.** Sweep `beta` and `nll_coef`. Or build the pairs from a different model than the one being trained (`data_path`).
- **GRPO.** Sweep `kl_coef` (try 0), `group_size` and `temperature`, or set `updates_per_batch` > 1 so the clipping matters. Or give addition partial credit per correct digit, or add a task in `tasks.py` with its own reward.
- **Chain the stages.** Run GRPO starting from the DPO model (`init_from`), or run DPO on pairs sampled from the GRPO model.

## Project layout

```
charlm/
  tokenizer.py    character tokenizer with chat (and reasoning) special tokens
  model.py        the transformer, sampling with a KV cache, per-token log-probabilities
  checkpoint.py   save/load a model together with its config and tokenizer
  config.py       dataclass configs from JSON files + key=value overrides
  training.py     shared training utilities (optimizer, LR schedule, precision, resuming, metrics logging)
  data.py         text files and batches
  datasets.py     downloading and cleaning larger corpora (`prepare-data`)
  chat.py         chat template, loss masking, batched reply sampling
  tasks.py        synthetic instruction tasks with held-out splits and a reward
  reasoning.py    the <|think|> reply format and the step-by-step arithmetic traces
  evaluation.py   task accuracy
  pretrain.py     stage 1
  sft.py          stage 2
  dpo.py          stage 3a
  grpo.py         stage 3b
  cli.py          `python -m charlm ...`
configs/
  example/        the example models (CPU)
  reasoning/      SFT and GRPO with and without reasoning (CPU)
  colab/          the 59M-parameter pipeline (GPU)
notebooks/colab_pipeline.ipynb   the GPU pipeline on Colab
.github/workflows/tests.yml   CI: lint and tests on pull requests and pushes to master
.github/workflows/data.yml    CI: checks that the dataset downloads still work (when the data code changes)
checkpoints/example/   trained example models, their metrics and the DPO pairs
  experiments/  metrics of the variants and comparison runs
docs/             README figures and the scripts that make them (make_figures.py, make_examples.py,
                  make_reasoning_figures.py, make_diagrams.py)
data/input.txt    Tiny Shakespeare (1.1M characters)
tests/            pytest suite, including a tiny end-to-end run of the pipeline
char_transformer_language_model.ipynb   the original self-contained notebook walkthrough
```

Run the tests with `pip install pytest && pytest tests`. GitHub Actions also runs them, together with the `pyflakes` linter, on every pull request and every push to `master` (`.github/workflows/tests.yml`).

## The original notebook

`char_transformer_language_model.ipynb` builds and trains the pretraining model step by step in a single file, without the library. It is a good place to start reading.

## License

This project is open source and available under the [MIT License](LICENSE).

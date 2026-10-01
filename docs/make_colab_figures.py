"""Draws the README figures of the 59M-parameter Colab run (notebooks/colab_pipeline.ipynb).

    pip install matplotlib
    python docs/make_colab_figures.py

The training logs are in docs/colab_run/ (copied from the run's checkpoints folder on Google Drive; the 240 MB
checkpoints are not in the repository).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (OUT, PX, ROOT, THEMES, axis_label, column, legend, line, new_figure,  # noqa: E402
                          panel_title, percent_axis, save, thousands)

RUN = os.path.join(ROOT, 'docs', 'colab_run')
ARITHMETIC = ['add', 'sub', 'mul', 'div']


def read(name):
    with open(os.path.join(RUN, f'{name}.metrics.jsonl')) as f:
        return [json.loads(line) for line in f]


def kept_step(rows):
    """ the step of the SFT checkpoint that was kept: the lowest validation loss """
    return min(rows, key=lambda r: r['val_loss'])['step']


def pretraining(t):
    rows = [r for r in read('base') if r['step'] > 0]  # step 0 (loss 4.78) would squash the rest
    hours = read('base')[-1]['time'] / 3600
    fig, (ax,) = new_figure(t, 'Pretraining the 59M model on TinyStories: the loss is still falling at the end',
                            f'Cross-entropy per character on training and held-out text, 60,000 steps of 64 x 512 '
                            f'characters ({hours:.1f} hours on a Colab GPU).')
    legend(fig, t, ['training loss', 'validation loss'], y=0.83)
    x, train = column(rows, 'train_loss')
    _, val = column(rows, 'val_loss')
    line(ax, t, x, train, t['series'][0])  # the two curves end almost together: only the held-out one is labelled
    line(ax, t, x, val, t['series'][1], f'val {val[-1]:.3f}')
    ax.set_ylim(0.3, 0.8)
    thousands(ax)
    axis_label(ax, t, x='training step', y='cross-entropy per character')
    return fig


def mean_arithmetic(rows):
    rows = [r for r in rows if all(f'acc/{k}' in r for k in ARITHMETIC)]
    return [r['step'] for r in rows], [100 * sum(r[f'acc/{k}'] for k in ARITHMETIC) / len(ARITHMETIC) for r in rows]


def finetuning(t):
    sft = read('sft')
    kept = kept_step(sft)
    stages = [('SFT with reasoning', sft), (f'GRPO (from SFT step {kept:,})', read('grpo')),
              (f'DPO (from SFT step {kept:,})', read('dpo'))]
    fig, axes = new_figure(t, 'Fine-tuning the 59M model: SFT works, GRPO and DPO make it worse',
                           'Held-out accuracy, 100 prompts per task. Arithmetic is the mean of add, sub, mul and '
                           'div.', ncols=3, height=3.4, top=0.64, wspace=0.3)
    fig.subplots_adjust(right=0.95)
    legend(fig, t, ['arithmetic', 'word problems (new phrasing)'], y=0.8)
    for ax, (title, rows) in zip(axes, stages):
        x, acc = mean_arithmetic(rows)
        line(ax, t, x, acc, t['series'][0], f'{acc[-1]:.0f}%')
        x, word = column(rows, 'acc/word', 100)
        line(ax, t, x, word, t['series'][1], f'{word[-1]:.0f}%')
        panel_title(ax, t, title)
        percent_axis(ax)
        ax.set_xlim(-0.03 * x[-1], 1.03 * x[-1])
        if x[-1] >= 2000:
            thousands(ax)
        axis_label(ax, t, x='step')
        if ax is not axes[0]:
            ax.set_yticklabels([])
    # the checkpoint SFT kept (lowest validation loss), which GRPO and DPO start from
    row = next(r for r in sft if r['step'] == kept)
    ax = axes[0]
    ax.axvline(kept, color=t['axis'], lw=1 * PX, zorder=1)
    ax.annotate(f'kept: step {kept:,}', (kept, 4), xytext=(-4, 0), textcoords='offset points', ha='right',
                va='bottom', color=t['ink2'], fontsize=8)
    ax.plot(kept, 100 * row['acc/word'], 'o', ms=8 * PX, mfc=t['series'][1], mec=t['surface'], mew=2 * PX, zorder=4,
            clip_on=False)
    return fig


def table():
    """ the README table: the kept SFT checkpoint and the final GRPO and DPO models """
    sft = read('sft')
    rows = [('SFT with reasoning (kept checkpoint)', next(r for r in sft if r['step'] == kept_step(sft))),
            ('+ GRPO', read('grpo')[-1]), ('+ DPO', read('dpo')[-1])]
    tasks = ['reverse', 'uppercase', 'spell', 'length', 'add', 'sub', 'mul', 'div', 'word']
    print('| model | ' + ' | '.join(tasks) + ' | overall |')
    print('|---|' + '---|' * (len(tasks) + 1))
    for name, r in rows:
        print(f'| {name} | ' + ' | '.join(f"{100 * r[f'acc/{k}']:.0f}%" for k in tasks) + f" | {100 * r['acc']:.1f}% |")
    base = read('base')
    print(f"\npretraining: {base[-1]['step']:,} steps, {base[-1]['time'] / 3600:.2f} h, final val loss "
          f"{base[-1]['val_loss']:.3f}, best {min(r['val_loss'] for r in base):.3f}")
    for name in ('sft', 'grpo', 'dpo'):
        print(f"{name}: {read(name)[-1]['time'] / 60:.1f} min of training")


def main():
    os.makedirs(OUT, exist_ok=True)
    table()
    for theme_name, t in THEMES.items():
        save(pretraining(t), 'colab_pretraining', theme_name)
        save(finetuning(t), 'colab_finetuning', theme_name)
    print(f"wrote figures to {OUT}")


if __name__ == '__main__':
    main()

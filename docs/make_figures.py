"""Draws the README figures from the example runs.

    pip install matplotlib
    python docs/make_figures.py

The training curves come from the *.metrics.jsonl logs in checkpoints/example/ (and experiments/ for the
variants that didn't work). The final comparison evaluates the example checkpoints on 500 held-out prompts per
task, which takes about a minute. Every figure is written twice, for GitHub's light and dark themes:
docs/figures/<name>.png and docs/figures/<name>_dark.png.
"""
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Patch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, 'checkpoints', 'example')
OUT = os.path.join(ROOT, 'docs', 'figures')
TASKS = ['reverse', 'uppercase', 'spell', 'length', 'add']

# colors per theme: chart surface, text inks, grid and axis hairlines, and three series colors
# (a colorblind-checked categorical palette, stepped separately for the dark surface)
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9',
                  axis='#c3c2b7', series=['#2a78d6', '#eb6834', '#1baf7a']),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a',
                 axis='#383835', series=['#3987e5', '#d95926', '#199e70']),
}
DPI = 100  # layout resolution: 1 unit below = 1 px; images are saved at 2x for sharp rendering
PX = 72 / DPI  # points per px


def read_metrics(path):
    with open(os.path.join(EXAMPLE, path)) as f:
        return [json.loads(line) for line in f]


def column(rows, key, scale=1.0):
    rows = [r for r in rows if key in r]
    return [r['step'] for r in rows], [r[key] * scale for r in rows]


# ---------------------------------------------------------------- shared styling

def new_figure(t, title, subtitle, ncols=1, height=3.6, width=8.0, top=0.74, wspace=0.28):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(t['surface'])
    fig.subplots_adjust(left=0.08, right=0.88, top=top, bottom=0.14, wspace=wspace)
    fig.text(0.02, 0.965, title, color=t['ink'], fontsize=12, fontweight='bold', va='top')
    fig.text(0.02, 0.965 - 0.3 / height, subtitle, color=t['ink2'], fontsize=9, va='top')
    for ax in axes[0]:
        ax.set_facecolor(t['surface'])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(t['axis'])
        ax.spines['bottom'].set_linewidth(1 * PX)
        ax.grid(axis='y', color=t['grid'], linewidth=1 * PX)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8.5, labelcolor=t['ink2'], pad=4)
    return fig, list(axes[0])


def panel_title(ax, t, text):
    ax.set_title(text, loc='left', color=t['ink'], fontsize=9.5, pad=8)


def axis_label(ax, t, x=None, y=None):
    if x:
        ax.set_xlabel(x, color=t['muted'], fontsize=8.5)
    if y:
        ax.set_ylabel(y, color=t['muted'], fontsize=8.5)


def legend(fig, t, labels, y=None, kind='line'):
    """ one row of legend keys under the subtitle; text in ink, identity in the key beside it """
    handles = [Line2D([], [], color=c, lw=2 * PX * 1.5, solid_capstyle='round') if kind == 'line' else
               Line2D([], [], ls='none', marker='o', ms=10 * PX, mfc=c, mec=t['surface'], mew=2 * PX) if kind == 'dot'
               else Patch(facecolor=c, edgecolor='none') for c in t['series'][:len(labels)]]
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.015, y), ncol=len(labels), frameon=False,
               fontsize=8.5, labelcolor=t['ink2'], handlelength=1.6, columnspacing=1.6, handletextpad=0.6)


def line(ax, t, x, y, color, end_label=None, dot=True):
    """ a 2px line with a ringed end-dot and an optional end label (in text ink, not the series color) """
    ax.plot(x, y, color=color, lw=2 * PX, solid_capstyle='round', solid_joinstyle='round', zorder=3)
    if dot:
        ax.plot(x[-1], y[-1], 'o', ms=10 * PX, mfc=color, mec=t['surface'], mew=2 * PX, zorder=4)
    if end_label:
        ax.annotate(end_label, (x[-1], y[-1]), xytext=(7, 0), textcoords='offset points', va='center',
                    color=t['ink2'], fontsize=8.5, annotation_clip=False)


def percent_axis(ax, top=100):
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 25))
    ax.set_yticklabels([f'{v}%' for v in range(0, top + 1, 25)])


def thousands(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))


def save(fig, name, theme_name):
    suffix = '' if theme_name == 'light' else '_dark'
    fig.savefig(os.path.join(OUT, f'{name}{suffix}.png'), dpi=2 * DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- the figures

def pretraining_loss(t):
    rows = [r for r in read_metrics('base.metrics.jsonl') if r['step'] > 0]  # step 0 (loss 4.63) squashes the rest
    fig, (ax,) = new_figure(t, 'Pretraining: loss on training and held-out text',
                            'Lower is better. Training stopped early at step 3,750; the best checkpoint '
                            '(step 3,000) became base.pt.')
    legend(fig, t, ['training loss', 'validation loss'], y=0.83)
    x, train = column(rows, 'train_loss')
    _, val = column(rows, 'val_loss')
    line(ax, t, x, train, t['series'][0], f'train {train[-1]:.2f}')
    line(ax, t, x, val, t['series'][1], f'val {val[-1]:.2f}')
    best = min(range(len(val)), key=val.__getitem__)
    ax.plot(x[best], val[best], 'o', ms=10 * PX, mfc=t['series'][1], mec=t['surface'], mew=2 * PX, zorder=4)
    ax.annotate(f'best {val[best]:.3f}', (x[best], val[best]), xytext=(0, 10), textcoords='offset points',
                ha='center', color=t['ink2'], fontsize=8.5)
    ax.set_ylim(1.0, 2.1)
    thousands(ax)
    axis_label(ax, t, x='training step', y='cross-entropy per character')
    return fig


def sft_accuracy(t):
    rows = read_metrics('sft.metrics.jsonl')
    fig, axes = new_figure(t, 'SFT: held-out accuracy per task during training',
                           'Measured on 50 held-out prompts per task. The word tasks reach 98-100% by step 2,000; '
                           'addition stays low.', ncols=5, height=3.0, top=0.7, wspace=0.35)
    fig.subplots_adjust(right=0.97)
    for ax, task in zip(axes, TASKS):
        x, acc = column(rows, f'acc/{task}', 100)
        line(ax, t, x, acc, t['series'][0])
        ax.annotate(f'{acc[-1]:.0f}%', (x[-1], acc[-1]), xytext=(0, 9 if acc[-1] < 90 else -14),
                    textcoords='offset points', ha='right', color=t['ink2'], fontsize=8.5)
        panel_title(ax, t, task)
        percent_axis(ax)
        ax.set_xticks([0, 1500, 3000])
        thousands(ax)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axis_label(axes[2], t, x='training step')
    return fig


def dpo_comparison(t):
    runs = [('DPO + NLL on chosen (nll_coef=1)', read_metrics('dpo.metrics.jsonl')),
            ('plain DPO (nll_coef=0)', read_metrics('experiments/dpo_plain.metrics.jsonl'))]
    fig, axes = new_figure(t, 'DPO: plain DPO makes the model worse; adding NLL on the chosen reply fixes it',
                           'Held-out accuracy during DPO (100 prompts per task), starting from the SFT model.',
                           ncols=2, top=0.66)
    legend(fig, t, [name for name, _ in runs], y=0.83)
    for ax, (key, title) in zip(axes, [('acc', 'overall accuracy'), ('acc/add', 'addition accuracy')]):
        for color, (_, rows) in zip(t['series'], runs):
            x, acc = column(rows, key, 100)
            line(ax, t, x, acc, color, f'{acc[-1]:.0f}%')
        panel_title(ax, t, title)
        axis_label(ax, t, x='DPO step')
    percent_axis(axes[0])
    percent_axis(axes[1], top=50)
    axes[1].set_yticks(range(0, 51, 10))
    axes[1].set_yticklabels([f'{v}%' for v in range(0, 51, 10)])
    return fig


def grpo_comparison(t):
    runs = [('add only, lr 1e-4, KL 0.04 (grpo.pt)', read_metrics('grpo.metrics.jsonl')),
            ('all tasks, groups of 8', read_metrics('experiments/grpo_all_tasks.metrics.jsonl')),
            ('add only, lr 3e-4, KL 0.01 (stopped)', read_metrics('experiments/grpo_lr3e-4_kl0.01.metrics.jsonl'))]
    fig, axes = new_figure(t, 'GRPO: training on the task with headroom, with a gentle update, is what works',
                           'Held-out addition accuracy (100 prompts) and the KL divergence from the SFT model '
                           'during GRPO.', ncols=2, top=0.66)
    legend(fig, t, [name for name, _ in runs], y=0.83)
    for color, (_, rows) in zip(t['series'], runs):
        x, acc = column(rows, 'acc/add', 100)
        line(axes[0], t, x, acc, color, f'{acc[-1]:.0f}%')
        x, kl = column(rows, 'kl')
        line(axes[1], t, x, kl, color, f'{kl[-1]:.2f}')
    panel_title(axes[0], t, 'addition accuracy')
    axes[0].set_ylim(0, 30)
    axes[0].set_yticks(range(0, 31, 10))
    axes[0].set_yticklabels([f'{v}%' for v in range(0, 31, 10)])
    panel_title(axes[1], t, 'KL divergence from the SFT model (per token)')
    axes[1].set_ylim(0, 1.0)
    for ax in axes:
        axis_label(ax, t, x='GRPO step')
    return fig


def evaluate_examples(n_per_task=500):
    """ accuracy of the example models on the held-out prompts, as in the README table """
    sys.path.insert(0, ROOT)
    from charlm import TaskSuite, evaluate_tasks, load_checkpoint
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    examples = suite.eval_set(n_per_task)
    results = {}
    for name in ('sft', 'dpo', 'grpo'):
        model, tokenizer, _ = load_checkpoint(os.path.join(EXAMPLE, f'{name}.pt'))
        results[name] = evaluate_tasks(model, tokenizer, examples)
        print(name, {k: round(v, 3) for k, v in results[name].items()})
    return results


def rounded_bar(ax, x0, width, height, color, radius_px=4):
    """ a bar with 4px rounded corners at its data end and square corners at the baseline """
    inv = ax.transData.inverted()
    (ox, oy), (px, py) = inv.transform([(0, 0), (1, 1)])
    rx, ry = radius_px * (px - ox), radius_px * (py - oy)
    ry = min(ry, height / 2)
    rx = min(rx, width / 2)
    x1 = x0 + width
    verts = [(x0, 0), (x0, height - ry), (x0, height), (x0 + rx, height), (x1 - rx, height), (x1, height),
             (x1, height - ry), (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none', zorder=3))


def final_comparison(t, results):
    groups = TASKS + ['overall']
    models = [('SFT', 'sft'), ('DPO', 'dpo'), ('GRPO', 'grpo')]
    fig, (ax,) = new_figure(t, 'The example models on 500 held-out prompts per task',
                            'SFT solves the word tasks; DPO and GRPO raise addition. The base model scores 0% '
                            'on all of them.')
    fig.subplots_adjust(right=0.97)
    legend(fig, t, [label for label, _ in models], y=0.83, kind='bar')
    ax.set_xlim(-0.5, len(groups) - 0.5)
    percent_axis(ax)
    fig.canvas.draw()  # fixes the layout, so pixel sizes can be converted to data units
    inv = ax.transData.inverted()
    px = inv.transform((1, 0))[0] - inv.transform((0, 0))[0]
    bar, gap = 24 * px, 2 * px  # 24px bars separated by a 2px surface gap
    total = len(models) * bar + (len(models) - 1) * gap
    for g, group in enumerate(groups):
        key = 'acc' if group == 'overall' else f'acc/{group}'
        for m, ((_, name), color) in enumerate(zip(models, t['series'])):
            value = results[name][key] * 100
            x0 = g - total / 2 + m * (bar + gap)
            rounded_bar(ax, x0, bar, value, color)
            if group == 'add':  # label only where the models clearly differ; the README table has every value
                ax.annotate(f'{value:.0f}%', (x0 + bar / 2, value), xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', color=t['ink2'], fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    results = evaluate_examples()
    for theme_name, t in THEMES.items():
        for name, draw in [('pretraining_loss', pretraining_loss), ('sft_accuracy', sft_accuracy),
                           ('dpo_comparison', dpo_comparison), ('grpo_comparison', grpo_comparison)]:
            save(draw(t), name, theme_name)
        save(final_comparison(t, results), 'final_comparison', theme_name)
    print(f"wrote figures to {OUT}")


if __name__ == '__main__':
    main()

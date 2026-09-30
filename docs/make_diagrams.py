"""Draws the README's explanatory diagrams: the map of the example models, and what SFT trains on.

    pip install matplotlib
    python docs/make_diagrams.py

Like the charts, each is written twice, for GitHub's light and dark themes (docs/figures/<name>[_dark].png).
The numbers in the model map are the held-out accuracies reported in the README.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import DPI, OUT, THEMES, save  # noqa: E402
from make_reasoning_figures import blend  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

WIDTH = 800  # px; the figures are laid out in pixels, y down


def canvas(t, height, title, subtitle):
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI), dpi=DPI)
    fig.patch.set_facecolor(t['surface'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.text(16, 14, title, color=t['ink'], fontsize=12, fontweight='bold', va='top')
    ax.text(16, 42, subtitle, color=t['ink2'], fontsize=9, va='top')
    return fig, ax


def box(ax, t, x, y, w, h, fill, radius=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=fill,
                                edgecolor='none', zorder=2))


# ---------------------------------------------------------------- the example models

NODES = {  # name: (x, y, title, lines, accent series index or None)
    'data': (16, 158, 'data/input.txt', ['Tiny Shakespeare', '1.1M characters'], None),
    'base': (222, 150, 'base.pt', ['writes Shakespeare', 'val loss 1.48', '0% on the tasks'], None),
    'sft': (428, 80, 'sft.pt', ['follows instructions', '82% on 5 tasks', 'addition 13%'], 0),
    'dpo': (634, 66, 'dpo.pt', ['82% on 5 tasks', 'addition 20%'], 0),
    'grpo': (634, 170, 'grpo.pt', ['84% on 5 tasks', 'addition 24%'], 0),
    'sft_r': (428, 262, 'sft_reasoning.pt', ['thinks step by step', '+ - * /: 99-100%', 'word problems 22%'], 1),
    'grpo_r': (634, 276, 'grpo_reasoning.pt', ['+ - * /: 99-100%', 'word problems 34%'], 1),
}
EDGES = [('data', 'base', 'pretrain'), ('base', 'sft', 'SFT'), ('sft', 'dpo', 'DPO'), ('sft', 'grpo', 'GRPO'),
         ('base', 'sft_r', 'SFT with reasoning'), ('sft_r', 'grpo_r', 'GRPO')]
NODE_W, LINE_H = 152, 15


def node_height(lines):
    return 30 + LINE_H * len(lines)


def model_map(t):
    fig, ax = canvas(t, 362, 'The example models in checkpoints/example/ and how each was made',
                     'Each arrow is one training stage. Accuracy is measured on held-out prompts of the tasks the '
                     'model was trained on.')
    fill = blend(t['grid'], t['surface'], 0.55)
    anchors = {}
    for name, (x, y, title, lines, accent) in NODES.items():
        h = node_height(lines)
        box(ax, t, x, y, NODE_W, h, fill)
        if accent is not None:  # the line of models: blue without reasoning, orange with it (as in the charts)
            box(ax, t, x, y, 4, h, t['series'][accent], radius=2)
        ax.text(x + 14, y + 11, title, color=t['ink'], fontsize=9, fontweight='bold', family='monospace', va='top')
        for i, text in enumerate(lines):
            ax.text(x + 14, y + 30 + i * LINE_H, text, color=t['ink2'], fontsize=8.5, va='top')
        anchors[name] = (x, y, h)
    for a, b, label in EDGES:
        ax_, ay, ah = anchors[a]
        bx, by, bh = anchors[b]
        start, end = (ax_ + NODE_W, ay + ah / 2), (bx, by + bh / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, color=t['muted'],
                                     lw=1.5, shrinkA=3, shrinkB=3, zorder=3))
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my - 4 if abs(end[1] - start[1]) < 40 else my, label, color=t['ink'], fontsize=8,
                ha='center', va='bottom' if abs(end[1] - start[1]) < 40 else 'center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=t['surface'], edgecolor='none'))
    return fig


# ---------------------------------------------------------------- what SFT trains on

PROMPT = 'What is 47 + 85?'
TRACE = '7+5+0=12 A=2, 4+8+1=13 A=132 => 132'
ANSWER = '132'


def token_row(ax, t, x, y, tokens, cell=15, height=26):
    """ draw tokens left to right from (x, y); each token is (text, kind): kind 'prompt', 'think' or 'answer';
    a text longer than one character is a special token, drawn as one wider cell. Returns the end x """
    fills = {'prompt': blend(t['grid'], t['surface'], 0.8), 'think': t['series'][1], 'answer': t['series'][0]}
    for text, kind in tokens:
        w = cell - 2 if len(text) == 1 else 9 + 6.2 * len(text)
        box(ax, t, x, y, w, height, fills[kind], radius=3)
        ink = t['ink'] if kind == 'prompt' else '#ffffff'
        ax.text(x + w / 2, y + height / 2, '␣' if text == ' ' else text, color=ink, ha='center',
                va='center', family='monospace', fontsize=8.5 if len(text) == 1 else 7.5,
                alpha=0.55 if text == ' ' else 1)
        x += w + 2  # a 2px surface gap between tokens
    return x


def sft_tokens(t):
    fig, ax = canvas(t, 230, 'What SFT trains on: every token of the reply, none of the prompt',
                     'One training example of a reasoning model, one character per token. Special tokens like '
                     '<|think|> are single tokens.')
    prompt = [('<|user|>', 'prompt')] + [(c, 'prompt') for c in PROMPT] + [('<|assistant|>', 'prompt')]
    reply = ([('<|think|>', 'think')] + [(c, 'think') for c in TRACE] + [('<|/think|>', 'think')]
             + [(c, 'answer') for c in ANSWER] + [('<|end|>', 'answer')])
    ax.text(16, 72, "the user's message: no loss (target -100)", color=t['ink2'], fontsize=8.5, va='top')
    token_row(ax, t, 16, 90, prompt)
    ax.text(16, 132, 'the reply: the model learns to predict each of these tokens, one after another',
            color=t['ink2'], fontsize=8.5, va='top')
    token_row(ax, t, 16, 150, reply, cell=14.2)
    # legend: what the colors mean
    fills = {'prompt': blend(t['grid'], t['surface'], 0.8), 'think': t['series'][1], 'answer': t['series'][0]}
    x = 16
    for kind, text in [('prompt', 'prompt (not trained on)'), ('think', 'reasoning (trained on)'),
                       ('answer', 'answer (trained on)')]:
        box(ax, t, x, 196, 12, 12, fills[kind], radius=3)
        ax.text(x + 18, 202, text, color=t['ink2'], fontsize=8.5, va='center')
        x += 36 + 6.3 * len(text)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for theme_name, t in THEMES.items():
        save(model_map(t), 'example_models', theme_name)
        save(sft_tokens(t), 'sft_tokens', theme_name)
    print(f"wrote diagrams to {OUT}")


if __name__ == '__main__':
    main()

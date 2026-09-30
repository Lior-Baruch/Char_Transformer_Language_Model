"""Draws the README figures and tables of the reasoning comparison: SFT and GRPO, each with and without reasoning.

    pip install matplotlib
    python docs/make_reasoning_figures.py

The models come from configs/reasoning/*.json. The plain (no reasoning) models are not in the repository, only
their metrics (checkpoints/example/experiments/); train them first to rerun the evaluation:

    python -m charlm sft  --config configs/reasoning/sft_plain.json
    python -m charlm grpo --config configs/reasoning/grpo_plain.json

The evaluation (500 held-out prompts per task, about 10 minutes on 4 CPU cores) is saved to
checkpoints/example/experiments/reasoning_eval.json and reused on later runs; delete it to evaluate again.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (EXAMPLE, OUT, ROOT, THEMES, axis_label, column, legend, line, new_figure,  # noqa: E402
                          panel_title, percent_axis, read_metrics, rounded_bar, save, thousands)

MODELS = [('SFT', 'experiments/sft_plain'), ('SFT + GRPO', 'experiments/grpo_plain'),
          ('SFT + reasoning', 'sft_reasoning'), ('SFT + reasoning + GRPO', 'grpo_reasoning')]
TASKS = ['reverse', 'uppercase', 'spell', 'length', 'add', 'sub', 'mul', 'div', 'word']
MATH = ['add', 'sub', 'mul', 'div', 'word']
EVAL_PER_TASK = 500
MAX_NEW_TOKENS = 100
RESULTS = os.path.join(EXAMPLE, 'experiments', 'reasoning_eval.json')


def evaluate():
    """ accuracy (and the replies) of the four models on EVAL_PER_TASK held-out prompts per task """
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    sys.path.insert(0, ROOT)
    from charlm import TaskSuite, evaluate_tasks, load_checkpoint
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    examples = suite.eval_set(EVAL_PER_TASK, TASKS)
    results = {'n_per_task': EVAL_PER_TASK, 'max_new_tokens': MAX_NEW_TOKENS,
               'distinct_prompts': {t: len({e.prompt for e in examples if e.task == t}) for t in TASKS},
               'models': {}}
    for label, path in MODELS:
        model, tokenizer, _ = load_checkpoint(os.path.join(EXAMPLE, f'{path}.pt'))
        metrics, replies = evaluate_tasks(model, tokenizer, examples, MAX_NEW_TOKENS, return_replies=True)
        results['models'][label] = {'metrics': metrics, 'replies': replies[::EVAL_PER_TASK // 50]}  # every 10th
        print(label, {k: round(v, 3) for k, v in metrics.items()})
    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=1)
    return results


def sft_curves(t):
    runs = [('SFT', read_metrics('experiments/sft_plain.metrics.jsonl')),
            ('SFT + reasoning', read_metrics('sft_reasoning.metrics.jsonl'))]
    fig, axes = new_figure(t, 'SFT with and without reasoning: held-out accuracy on the math tasks',
                           'Measured every 250 steps on 50 held-out prompts per task. Same data, steps and '
                           'batch size; only the replies differ.', ncols=5, height=3.2, top=0.66, wspace=0.35)
    fig.subplots_adjust(right=0.97)
    legend(fig, t, [name for name, _ in runs], y=0.8)
    for ax, task in zip(axes, MATH):
        for color, (_, rows) in zip(t['series'], runs):
            x, acc = column(rows, f'acc/{task}', 100)
            line(ax, t, x, acc, color)
        panel_title(ax, t, task)
        percent_axis(ax)
        thousands(ax)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axis_label(axes[2], t, x='training step')
    return fig


def grpo_curves(t):
    runs = [('SFT + GRPO', read_metrics('experiments/grpo_plain.metrics.jsonl')),
            ('SFT + reasoning + GRPO', read_metrics('grpo_reasoning.metrics.jsonl'))]
    fig, axes = new_figure(t, 'GRPO on the math tasks, with and without reasoning',
                           'Left: the share of correct sampled replies in each step\'s groups. Right: held-out '
                           'accuracy on the five math tasks (50 prompts each).', ncols=2, top=0.66)
    legend(fig, t, [name for name, _ in runs], y=0.83)
    for color, (_, rows) in zip(t['series'], runs):
        x, reward = column(rows, 'reward', 100)
        line(axes[0], t, x, reward, color, f'{reward[-1]:.0f}%')
        math_rows = [r for r in rows if all(f'acc/{k}' in r for k in MATH)]
        x = [r['step'] for r in math_rows]
        acc = [100 * sum(r[f'acc/{k}'] for k in MATH) / len(MATH) for r in math_rows]
        line(axes[1], t, x, acc, color, f'{acc[-1]:.0f}%')
    panel_title(axes[0], t, 'reward on sampled replies')
    panel_title(axes[1], t, 'math accuracy (held-out, greedy)')
    for ax in axes:
        percent_axis(ax)
        axis_label(ax, t, x='GRPO step')
    return fig


def blend(color, surface, amount):
    """ color mixed with the chart surface: a lighter (or, on dark, dimmer) shade of the same hue """
    c, s = (tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in (color, surface))
    return '#' + ''.join(f'{round(a * amount + b * (1 - amount)):02x}' for a, b in zip(c, s))


def final_bars(t, results):
    groups = MATH + ['all 9 tasks']
    labels = [label for label, _ in MODELS]
    # one hue per arm (plain, reasoning); the SFT model in a light shade, the model after GRPO in the full color
    series = [blend(t['series'][0], t['surface'], 0.45), t['series'][0],
              blend(t['series'][1], t['surface'], 0.45), t['series'][1]]
    fig, (ax,) = new_figure(t, f'Math accuracy of the four models on {EVAL_PER_TASK} held-out prompts per task',
                            'Word problems are asked in a phrasing never seen in training. "all 9 tasks" '
                            'includes the four word tasks.')
    fig.subplots_adjust(right=0.97)
    legend(fig, dict(t, series=series), labels, y=0.83, kind='bar')
    ax.set_xlim(-0.5, len(groups) - 0.5)
    percent_axis(ax)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    px = inv.transform((1, 0))[0] - inv.transform((0, 0))[0]
    bar, gap = 18 * px, 2 * px
    total = len(labels) * bar + (len(labels) - 1) * gap
    for g, group in enumerate(groups):
        key = 'acc' if group == 'all 9 tasks' else f'acc/{group}'
        for m, (label, color) in enumerate(zip(labels, series)):
            value = results['models'][label]['metrics'][key] * 100
            x0 = g - total / 2 + m * (bar + gap)
            rounded_bar(ax, x0, bar, value, color)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    return fig


def cell(text):
    return '`' + text.replace('|', '\\|') + '`' if text else '*(empty)*'


def tables(results):
    """ the README's accuracy table and example replies """
    labels = [label for label, _ in MODELS]
    print('| model | ' + ' | '.join(TASKS) + ' | overall |')
    print('|---|' + '---|' * (len(TASKS) + 1))
    for label in labels:
        m = results['models'][label]['metrics']
        print(f'| {label} | ' + ' | '.join(f"{100 * m[f'acc/{t}']:.1f}%" for t in TASKS)
              + f" | {100 * m['acc']:.1f}% |")
    print('\nreasoning matches the taught method:')
    for label in labels:
        m = results['models'][label]['metrics']
        print(label, {t: round(m[f'trace/{t}'], 3) for t in MATH if f'trace/{t}' in m})
    print('\ndistinct prompts per task:', results['distinct_prompts'])


def examples(results, per_task=2):
    """ the first held-out prompts of each math task and the replies of the SFT models """
    sys.path.insert(0, ROOT)
    from charlm import TaskSuite, score
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    eval_set = suite.eval_set(EVAL_PER_TASK, TASKS)[::EVAL_PER_TASK // 50]  # the prompts whose replies were kept
    shown = [(i, e) for task in MATH for i, e in [(i, e) for i, e in enumerate(eval_set) if e.task == task][:per_task]]
    print('\n| prompt | expected | SFT | SFT + reasoning |')
    print('|---|---|---|---|')
    for i, e in shown:
        replies = [results['models'][label]['replies'][i] for label in ('SFT', 'SFT + reasoning')]
        marks = [f"{cell(r)} {'✓' if score(e, r) else '✗'}" for r in replies]
        print(f'| {e.prompt} | {cell(e.answer)} | ' + ' | '.join(marks) + ' |')


def main():
    os.makedirs(OUT, exist_ok=True)
    results = evaluate()
    tables(results)
    examples(results)
    for theme_name, t in THEMES.items():
        save(sft_curves(t), 'reasoning_sft', theme_name)
        save(grpo_curves(t), 'reasoning_grpo', theme_name)
        save(final_bars(t, results), 'reasoning_final', theme_name)
    print(f"wrote figures to {OUT}")


if __name__ == '__main__':
    main()

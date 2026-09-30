"""Draws the README figures and tables of the reasoning comparison: SFT and GRPO, each with and without reasoning.

    pip install matplotlib
    python docs/make_reasoning_figures.py

The models come from configs/reasoning/*.json. The plain (no reasoning) models are not in the repository, only
their metrics (checkpoints/example/experiments/); train them first to rerun the evaluation:

    python -m charlm sft  --config configs/reasoning/sft_plain.json
    python -m charlm grpo --config configs/reasoning/grpo_plain.json

The evaluation (500 held-out prompts per task, a few minutes per model on a CPU) is saved to
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


def evaluate(models=MODELS):
    """ accuracy (and every 10th reply) of the models on EVAL_PER_TASK held-out prompts per task; each model's
    results are saved as soon as they are computed, and models already in the file are not evaluated again """
    results = {'n_per_task': EVAL_PER_TASK, 'max_new_tokens': MAX_NEW_TOKENS, 'models': {}}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    missing = [(label, path) for label, path in models if label not in results['models']]
    if not missing:
        return results
    sys.path.insert(0, ROOT)
    from charlm import TaskSuite, evaluate_tasks, load_checkpoint
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    examples = suite.eval_set(EVAL_PER_TASK, TASKS)
    results['distinct_prompts'] = {t: len({e.prompt for e in examples if e.task == t}) for t in TASKS}
    for label, path in missing:
        if not os.path.exists(os.path.join(EXAMPLE, f'{path}.pt')):
            raise FileNotFoundError(f"{path}.pt is missing; train it first (see the top of this file)")
        model, tokenizer, _ = load_checkpoint(os.path.join(EXAMPLE, f'{path}.pt'))
        metrics, replies = evaluate_tasks(model, tokenizer, examples, MAX_NEW_TOKENS, return_replies=True)
        results['models'][label] = {'metrics': metrics, 'replies': replies[::EVAL_PER_TASK // 50]}  # every 10th
        print(label, {k: round(v, 3) for k, v in metrics.items()}, flush=True)
        with open(RESULTS, 'w') as f:
            json.dump(results, f, indent=1)
    return results


def sft_curves(t):
    runs = [('SFT', read_metrics('experiments/sft_plain.metrics.jsonl')),
            ('SFT + reasoning', read_metrics('sft_reasoning.metrics.jsonl'))]
    fig, axes = new_figure(t, 'SFT with and without reasoning: held-out accuracy on the math tasks',
                           'Measured every 250 steps on 50 held-out prompts per task. Same data, steps and '
                           'batch size; only the replies differ.', ncols=5, height=3.3, top=0.62, wspace=0.35)
    fig.subplots_adjust(right=0.97)
    legend(fig, t, [name for name, _ in runs], y=0.79)
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
                           'Training reward, groups with both right and wrong replies (the only ones that teach '
                           'anything), and held-out word problems.', ncols=3, height=3.4,
                           top=0.64, wspace=0.45)
    fig.subplots_adjust(right=0.93)
    legend(fig, t, [name for name, _ in runs], y=0.8)
    for color, (_, rows) in zip(t['series'], runs):
        x, reward = column(rows, 'reward', 100)
        line(axes[0], t, x, reward, color, f'{reward[-1]:.0f}%')
        x, signal = column(rows, 'groups_with_signal', 100)
        line(axes[1], t, x, signal, color, f'{signal[-1]:.0f}%')
        x, acc = column(rows, 'acc/word', 100)
        line(axes[2], t, x, acc, color, f'{acc[-1]:.0f}%')
    panel_title(axes[0], t, 'reward (correct sampled replies)')
    panel_title(axes[1], t, 'groups with a learning signal')
    panel_title(axes[2], t, 'word problems, new phrasing')
    for ax in axes:
        percent_axis(ax)
        axis_label(ax, t, x='GRPO step')
    return fig


def blend(color, other, amount):
    """ color mixed with another color: amount 1 is color, 0 is other """
    c, s = (tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in (color, other))
    return '#' + ''.join(f'{round(a * amount + b * (1 - amount)):02x}' for a, b in zip(c, s))


def tint(t, color):
    """ a lighter shade of a series color, for the "before" state: mixed with the surface on the light theme, and
    with white on the dark one (mixing with a dark surface would make it too dim to see) """
    return blend(color, t['surface'], 0.45) if t is THEMES['light'] else blend(color, '#ffffff', 0.5)


def final_bars(t, results):
    groups = MATH + ['all 9 tasks']
    labels = [label for label, _ in MODELS]
    # one hue per arm (plain, reasoning); the SFT model in a light shade, the model after GRPO in the full color
    series = [tint(t, t['series'][0]), t['series'][0], tint(t, t['series'][1]), t['series'][1]]
    fig, (ax,) = new_figure(t, f'Math accuracy of the four models on {EVAL_PER_TASK} held-out prompts per task',
                            'Word problems use a phrasing never seen in training. "all 9 tasks" adds reverse, '
                            'uppercase, spell and length.')
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


def equation_value(reasoning):
    """ the result of the equation a word problem's reasoning starts with ("12/3: ..." -> 4), or None """
    import re
    match = re.match(r'(\d+)([-+*/])(\d+):', reasoning)
    if not match:
        return None
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    if op == '/':
        return a // b if b and a % b == 0 else None
    return {'+': a + b, '-': a - b, '*': a * b}[op]


def cached(key, label, compute):
    """ an analysis of one model, saved in the results file (so the figures can be redrawn without the models) """
    results = evaluate()
    if label not in results.setdefault(key, {}):
        results[key][label] = compute(label)
        with open(RESULTS, 'w') as f:
            json.dump(results, f, indent=1)
    return results[key][label]


def load_model(label):
    sys.path.insert(0, ROOT)
    from charlm import load_checkpoint
    path = os.path.join(EXAMPLE, f'{dict(MODELS)[label]}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing; train it first (see the top of this file)")
    return load_checkpoint(path)[:2]


def eval_examples(tasks):
    sys.path.insert(0, ROOT)
    from charlm import TaskSuite
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    return [e for e in suite.eval_set(EVAL_PER_TASK, TASKS) if e.task in tasks]


def word_phrasings(label='SFT + reasoning', quiet=False):
    """ the evaluation's word problems (held-out numbers), asked in the held-out phrasing and in a training phrasing:
    accuracy, how often the equation the model writes first has the right result (i.e. it read the story right,
    whether or not it then computes it right), and the accuracy per operation """
    def compute(label):
        import random
        import re
        from charlm import sample_replies, score
        from charlm.reasoning import split_reply
        from charlm.tasks import ITEMS, NAMES, WORD_TEMPLATES, Example
        model, tokenizer = load_model(label)
        held_out = eval_examples(['word'])
        rng = random.Random(0)
        equation = lambda e: re.match(r'(\d+)([-+*/])(\d+)', e.reasoning).groups()
        trained = [Example('word', rng.choice(WORD_TEMPLATES[op][:-1]).format(
            n=rng.choice(NAMES), i=rng.choice(ITEMS), a=a, b=b), e.answer, e.reasoning)
            for e in held_out for a, op, b in [equation(e)]]
        out = {}
        for name, examples in [('new', held_out), ('trained', trained)]:
            replies = [r[0][1] for r in sample_replies(model, tokenizer, [e.prompt for e in examples],
                                                       max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)]
            right = [score(e, r) for e, r in zip(examples, replies)]
            equations = [equation_value(split_reply(r)[0]) == int(e.answer) for e, r in zip(examples, replies)]
            per_op = {op: [x for x, e in zip(right, examples) if equation(e)[1] == op] for op in '+-*/'}
            out[name] = {'acc': sum(right) / len(right), 'equation': sum(equations) / len(equations),
                         'per_op': {op: sum(v) / len(v) for op, v in per_op.items()},
                         'n_per_op': {op: len(v) for op, v in per_op.items()}}
        return out
    result = cached('word_phrasings', label, compute)
    if quiet:
        return result
    print(f'\n{label}: word problems with held-out numbers')
    print('| phrasing | accuracy | reads the story right | + | - | * | / |')
    print('|---|---|---|---|---|---|---|')
    for name, title in [('new', 'never trained on'), ('trained', 'trained on')]:
        r = result[name]
        print(f"| {title} | {100 * r['acc']:.0f}% | {100 * r['equation']:.0f}% | "
              + ' | '.join(f"{100 * r['per_op'][op]:.0f}%" for op in '+-*/') + ' |')
    return result


PLACES = ['ten-thousands', 'thousands', 'hundreds', 'tens', 'ones']
SHORT_PLACES = {'ten-thousands': '10000s', 'thousands': '1000s', 'hundreds': '100s', 'tens': '10s', 'ones': '1s'}


def digit_accuracy(label):
    """ for each arithmetic task, how often each digit of the answer is right, by place value (for answers that
    have that digit); the plain model's mistakes are not spread evenly """
    def compute(label):
        from charlm import sample_replies
        from charlm.reasoning import answer_of
        model, tokenizer = load_model(label)
        examples = eval_examples(['add', 'sub', 'mul', 'div'])
        replies = [r[0][1] for r in sample_replies(model, tokenizer, [e.prompt for e in examples],
                                                   max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)]
        out = {}
        for task in ('add', 'sub', 'mul', 'div'):
            right, total = [0] * 5, [0] * 5
            for e, reply in zip(examples, replies):
                if e.task != task:
                    continue
                answer = answer_of(reply).strip()
                for i in range(len(e.answer)):  # i = 0 is the ones digit
                    total[i] += 1
                    right[i] += len(answer) > i and answer[-1 - i] == e.answer[-1 - i]
            out[task] = {PLACES[-1 - i]: {'acc': right[i] / total[i], 'n': total[i]} for i in range(5) if total[i]}
        return out
    return cached('digits', label, compute)


def reply_tokens(results, label, task):
    """ mean reply length in tokens (the <|think|> markers count as one token each, plus <|end|>) of the kept
    replies of one task """
    import re
    replies = [r for r, e in zip(results['models'][label]['replies'], kept_tasks()) if e == task]
    lengths = [len(re.sub(r'<\|/?think\|>', '#', r)) + 1 for r in replies]
    return sum(lengths) / len(lengths)


def kept_tasks():
    """ the task of each kept reply (every 10th of the evaluation set) """
    return [t for t in TASKS for _ in range(EVAL_PER_TASK)][::EVAL_PER_TASK // 50]


# ---------------------------------------------------------------- the analysis figures

def digits_figure(t):
    plain, reasoning = digit_accuracy('SFT'), digit_accuracy('SFT + reasoning')
    fig, axes = new_figure(t, 'Without reasoning, the leading digit is usually right and the others often wrong',
                           'Share of held-out answers with each digit right, by place value (500 prompts per task). '
                           'Answers are written left to right.', ncols=4, height=3.4, top=0.64, wspace=0.18)
    fig.subplots_adjust(right=0.98, left=0.07)
    legend(fig, t, ['SFT', 'SFT + reasoning'], y=0.8, kind='bar')
    fig.canvas.draw()
    names = {'add': 'add (up to 99 + 99)', 'sub': 'sub (up to 999 - 999)', 'mul': 'mul (up to 999 x 9)',
             'div': 'div (up to 8991 / 9)'}
    for ax, task in zip(axes, ('add', 'sub', 'mul', 'div')):
        places = [p for p in PLACES if p in plain[task]]
        ax.set_xlim(-0.6, len(places) - 0.4)
        percent_axis(ax)
        fig.canvas.draw()
        inv = ax.transData.inverted()
        px = inv.transform((1, 0))[0] - inv.transform((0, 0))[0]
        bar, gap = 14 * px, 2 * px
        for i, place in enumerate(places):
            for m, (data, color) in enumerate([(plain, t['series'][0]), (reasoning, t['series'][1])]):
                value = 100 * data[task][place]['acc']
                x0 = i - bar - gap / 2 + m * (bar + gap)
                rounded_bar(ax, x0, bar, value, color, radius_px=3)
                if m == 0:  # label the plain model's bars; the reasoning model's are all 98-100%
                    ax.annotate(f'{value:.0f}', (x0 + bar / 2, value), xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', color=t['ink2'], fontsize=7.5)
        ax.set_ylim(0, 110)  # room for the labels above full bars
        ax.set_xticks(range(len(places)))
        ax.set_xticklabels([SHORT_PLACES[p] for p in places], fontsize=8.5)
        panel_title(ax, t, names[task])
        if ax is not axes[0]:
            ax.set_yticklabels([])
    return fig


def cost_figure(t, results):
    """ reply length against accuracy: what the reasoning costs and what it buys, per math task """
    rows = MATH[::-1]  # top to bottom: add ... word
    runs = [('SFT', t['series'][0]), ('SFT + reasoning', t['series'][1])]
    fig, (ax,) = new_figure(t, 'Reasoning costs ~15x longer replies, and buys the arithmetic',
                            'Average reply length in tokens (bars) and held-out accuracy (labels), per math task.',
                            height=3.6, top=0.72)
    fig.subplots_adjust(left=0.1, right=0.97)
    legend(fig, t, [label for label, _ in runs], y=0.85, kind='bar')
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=t['grid'], linewidth=1 * 72 / 100)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(t['axis'])
    ax.spines['left'].set_linewidth(72 / 100)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    py = inv.transform((0, 1))[1] - inv.transform((0, 0))[1]
    height, gap = 12 * abs(py), 2 * abs(py)
    for r, task in enumerate(rows):
        for m, (label, color) in enumerate(runs):
            length = reply_tokens(results, label, task)
            acc = 100 * results['models'][label]['metrics'][f'acc/{task}']
            y0 = r + gap / 2 - m * (height + gap)  # the plain model's bar above the reasoning model's
            horizontal_bar(ax, y0, height, length, color)
            ax.annotate(f'{length:.0f} tokens, {acc:.0f}% right', (length, y0 + height / 2), xytext=(5, 0),
                        textcoords='offset points', va='center', color=t['ink2'], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis='y', labelsize=9, labelcolor=t['ink'])
    axis_label(ax, t, x='reply length (tokens)')
    return fig


def horizontal_bar(ax, y0, height, length, color, radius_px=4):
    """ a bar from x=0 with 4px rounded corners at its data end and square corners at the baseline """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    inv = ax.transData.inverted()
    (ox, oy), (px, py) = inv.transform([(0, 0), (1, 1)])
    rx, ry = min(radius_px * (px - ox), length / 2), min(radius_px * abs(py - oy), height / 2)
    y1 = y0 + height
    verts = [(0, y0), (length - rx, y0), (length, y0), (length, y0 + ry), (length, y1 - ry), (length, y1),
             (length - rx, y1), (0, y1), (0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none', zorder=3))


def word_figure(t):
    """ word problems per operation: the phrasings the model trained on vs the one it never saw """
    sft, grpo = word_phrasings('SFT + reasoning', quiet=True), word_phrasings('SFT + reasoning + GRPO', quiet=True)
    ops = [('+', 'addition'), ('-', 'subtraction'), ('*', 'multiplication'), ('/', 'division')]
    series = [('trained phrasing (SFT + reasoning)', t['muted'], sft['trained']),
              ('new phrasing, SFT + reasoning', tint(t, t['series'][1]), sft['new']),
              ('new phrasing, after GRPO', t['series'][1], grpo['new'])]
    fig, (ax,) = new_figure(t, 'Word problems: right in the phrasings it trained on, mostly wrong in a new one',
                            'Held-out word problems (the same numbers in both phrasings), by the operation the story '
                            'needs.', height=3.4, top=0.7)
    fig.subplots_adjust(left=0.15, right=0.95)
    legend(fig, dict(t, series=[c for _, c, _ in series]), [name for name, _, _ in series], y=0.84, kind='dot')
    ax.set_ylim(-0.6, len(ops) - 0.4)
    ax.set_xlim(0, 100)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=t['grid'], linewidth=72 / 100)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f'{v}%' for v in range(0, 101, 25)])
    for r, (op, name) in enumerate(ops[::-1]):
        values = [100 * data['per_op'][op] for _, _, data in series]
        ax.plot([min(values), max(values)], [r, r], color=t['grid'], lw=3 * 72 / 100, solid_capstyle='round',
                zorder=2)
        for (_, color, _), value in zip(series, values):
            ax.plot(value, r, 'o', ms=11 * 72 / 100, mfc=color, mec=t['surface'], mew=2 * 72 / 100, zorder=3)
        ax.annotate(f'{values[1]:.0f}% \u2192 {values[2]:.0f}%', (min(values[1:]), r), xytext=(0, -13),
                    textcoords='offset points', ha='center', color=t['ink2'], fontsize=8)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([name for _, name in ops[::-1]])
    ax.tick_params(axis='y', labelsize=9, labelcolor=t['ink'])
    return fig


def main():
    if sys.argv[1:2] == ['--only']:  # evaluate some models now, e.g. while another one is still training
        evaluate([m for m in MODELS if m[0] in sys.argv[2:]])
        return
    os.makedirs(OUT, exist_ok=True)
    results = evaluate()
    tables(results)
    examples(results)
    for label in ('SFT + reasoning', 'SFT + reasoning + GRPO'):
        word_phrasings(label)
    for label in ('SFT', 'SFT + reasoning'):
        digit_accuracy(label)
    results = evaluate()  # now with the analyses
    for theme_name, t in THEMES.items():
        save(sft_curves(t), 'reasoning_sft', theme_name)
        save(grpo_curves(t), 'reasoning_grpo', theme_name)
        save(final_bars(t, results), 'reasoning_final', theme_name)
        save(digits_figure(t), 'reasoning_digits', theme_name)
        save(cost_figure(t, results), 'reasoning_cost', theme_name)
        save(word_figure(t), 'reasoning_word', theme_name)
    print(f"wrote figures to {OUT}")


if __name__ == '__main__':
    main()

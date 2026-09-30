"""Prints the example tables in the README: the example models' replies to held-out prompts, "Say a line as ..."
replies, and a few DPO preference pairs. All replies are greedy, so they are the same every run.

    python docs/make_examples.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from charlm import TaskSuite, load_checkpoint, sample_replies, score  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'checkpoints', 'example')
MODELS = ['sft', 'dpo', 'grpo']
EVAL_PER_TASK = 500  # the evaluation set behind the README table; its prompts depend on this size
PER_TASK = {'reverse': 2, 'uppercase': 1, 'spell': 2, 'length': 1, 'add': 5}  # the first prompts of each task
SPEAKERS = ['ROMEO', 'JULIET', 'KING RICHARD III', 'Nurse']


def cell(text):
    return '`' + text.replace('|', '\\|').replace('\n', '↵') + '`' if text else '*(empty)*'


def main():
    suite = TaskSuite(open(os.path.join(ROOT, 'data', 'input.txt')).read())
    eval_set = suite.eval_set(EVAL_PER_TASK)
    examples = [e for task, n in PER_TASK.items() for e in [x for x in eval_set if x.task == task][:n]]
    models = {name: load_checkpoint(os.path.join(EXAMPLE, f'{name}.pt'))[:2] for name in MODELS}

    replies = {name: [r[0][1] for r in sample_replies(model, tok, [e.prompt for e in examples], temperature=0.0)]
               for name, (model, tok) in models.items()}
    print('| prompt | expected | ' + ' | '.join(n.upper() for n in MODELS) + ' |')
    print('|---|---|' + '---|' * len(MODELS))
    for i, e in enumerate(examples):
        marks = [f"{cell(replies[n][i])} {'✓' if score(e, replies[n][i]) else '✗'}" for n in MODELS]
        print(f'| {e.prompt} | {cell(e.answer)} | ' + ' | '.join(marks) + ' |')

    print()
    prompts = [f'Say a line as {s}.' for s in SPEAKERS]
    lines = {name: [r[0][1] for r in sample_replies(model, tok, prompts, temperature=0.0)]
             for name, (model, tok) in models.items() if name in ('sft', 'grpo')}
    print('| prompt | SFT | GRPO |')
    print('|---|---|---|')
    for i, p in enumerate(prompts):
        print(f"| {p} | {lines['sft'][i]} | {lines['grpo'][i]} |")

    print()
    with open(os.path.join(EXAMPLE, 'dpo.pairs.jsonl')) as f:
        pairs = [json.loads(line) for line in f]
    shown = [p for p in pairs if p['task'] == 'add'][:3] + [p for p in pairs if p['task'] != 'add'][:3]
    print('| prompt | chosen (correct answer) | rejected (the SFT model\'s own sampled answer) |')
    print('|---|---|---|')
    for p in shown:
        print(f"| {p['prompt']} | {cell(p['chosen'])} | {cell(p['rejected'])} |")


if __name__ == '__main__':
    main()

"""Measuring how often a model answers task prompts correctly."""
from collections import defaultdict

from .chat import sample_replies
from .tasks import score


def evaluate_tasks(model, tokenizer, examples, max_new_tokens=32, return_replies=False):
    """ greedy-decode a reply to every example; returns {'acc': overall accuracy, 'acc/<task>': per-task accuracy}
    (and the replies, if return_replies) """
    replies = [r[0][1] for r in sample_replies(model, tokenizer, [e.prompt for e in examples],
                                               max_new_tokens=max_new_tokens, temperature=0.0)]
    correct, total = defaultdict(float), defaultdict(int)
    for example, reply in zip(examples, replies):
        correct[example.task] += score(example, reply)
        total[example.task] += 1
    results = {'acc': sum(correct.values()) / max(1, len(examples))}
    results.update({f'acc/{task}': correct[task] / total[task] for task in total})
    return (results, replies) if return_replies else results

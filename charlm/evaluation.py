"""Measuring how often a model answers task prompts correctly."""
from collections import defaultdict

from .chat import sample_replies
from .reasoning import split_reply
from .tasks import score
from .tokenizer import THINK


def evaluate_tasks(model, tokenizer, examples, max_new_tokens=32, return_replies=False):
    """ greedy-decode a reply to every example; returns {'acc': overall accuracy, 'acc/<task>': per-task accuracy}
    (and the replies, if return_replies). When the model reasons, it also returns 'trace/<task>': the share of
    replies whose reasoning is exactly the task's step-by-step trace, i.e. whether the model follows the method """
    replies = [r[0][1] for r in sample_replies(model, tokenizer, [e.prompt for e in examples],
                                               max_new_tokens=max_new_tokens, temperature=0.0)]
    correct, total, traced = defaultdict(float), defaultdict(int), defaultdict(float)
    reasons = any(THINK in reply for reply in replies)
    for example, reply in zip(examples, replies):
        correct[example.task] += score(example, reply)
        total[example.task] += 1
        if reasons and example.reasoning:
            traced[example.task] += split_reply(reply)[0].strip() == example.reasoning
    results = {'acc': sum(correct.values()) / max(1, len(examples))}
    results.update({f'acc/{task}': correct[task] / total[task] for task in total})
    if reasons:
        results.update({f'trace/{task}': traced[task] / total[task] for task in total
                        if any(e.reasoning for e in examples if e.task == task)})
    return (results, replies) if return_replies else results

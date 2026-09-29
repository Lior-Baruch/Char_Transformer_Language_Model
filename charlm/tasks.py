"""Synthetic instruction-following tasks built from the pretraining corpus.

Every verifiable task has a single correct answer, so replies can be scored automatically. That gives
labeled data for SFT, correct/incorrect reply pairs for DPO and a reward for GRPO. The "speak" task asks for
a line in the style of a character from the corpus; it has no single right answer and is only used for SFT.

Words, number pairs and speeches are split into a train part and a held-out eval part, so evaluation
measures whether the model learned the task rather than memorized the training examples.
"""
import random
import re
from dataclasses import asdict, dataclass

VERIFIABLE_TASKS = ('reverse', 'uppercase', 'spell', 'length', 'add')
ALL_TASKS = VERIFIABLE_TASKS + ('speak',)


@dataclass
class Example:
    task: str
    prompt: str
    answer: str

    def to_dict(self):
        return asdict(self)


def score(example, reply):
    """ reward for a reply: 1.0 if it is exactly the expected answer (ignoring surrounding whitespace), else 0.0 """
    return 1.0 if reply.strip() == example.answer else 0.0


def parse_speeches(text, min_speeches=20, min_len=10, max_len=60):
    """ (speaker, first line of the speech) for speakers with at least min_speeches speeches """
    speeches = []
    for block in text.split('\n\n'):
        lines = block.strip().split('\n')
        if len(lines) > 1 and re.fullmatch(r"[A-Za-z][A-Za-z' ]*:", lines[0]):
            line = lines[1].strip()
            if min_len <= len(line) <= max_len:
                speeches.append((lines[0][:-1], line))
    counts = {}
    for speaker, _ in speeches:
        counts[speaker] = counts.get(speaker, 0) + 1
    return [(s, line) for s, line in speeches if counts[s] >= min_speeches]


class TaskSuite:
    """ generates task examples; the train/eval split is fixed by split_seed """

    def __init__(self, text, split_seed=0, eval_fraction=0.2, min_word_len=3, max_word_len=8, max_number=99):
        rng = random.Random(split_seed)
        words = sorted({w for w in re.findall(r"[a-z]+", text.lower()) if min_word_len <= len(w) <= max_word_len})
        pairs = [(a, b) for a in range(max_number + 1) for b in range(max_number + 1)]
        speeches = parse_speeches(text)
        self.pools = {}
        for name, items in (('words', words), ('pairs', pairs), ('speeches', speeches)):
            items = list(items)
            rng.shuffle(items)
            n_eval = int(len(items) * eval_fraction)
            self.pools[name] = {'eval': items[:n_eval], 'train': items[n_eval:]}

    def make(self, task, rng, split='train'):
        """ one random example of the given task """
        word = lambda: rng.choice(self.pools['words'][split])
        if task == 'reverse':
            w = word()
            return Example(task, f"Reverse the word: {w}", w[::-1])
        if task == 'uppercase':
            w = word()
            return Example(task, f"Write in capital letters: {w}", w.upper())
        if task == 'spell':
            w = word()
            return Example(task, f"Spell out: {w}", '-'.join(w))
        if task == 'length':
            w = word()
            return Example(task, f'How many letters are in "{w}"?', str(len(w)))
        if task == 'add':
            a, b = rng.choice(self.pools['pairs'][split])
            return Example(task, f"What is {a} + {b}?", str(a + b))
        if task == 'speak':
            speaker, line = rng.choice(self.pools['speeches'][split])
            return Example(task, f"Say a line as {speaker}.", line)
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(ALL_TASKS)}")

    def sample(self, n, tasks=ALL_TASKS, split='train', seed=0):
        """ n examples, each of a task picked uniformly at random """
        rng = random.Random(seed)
        return [self.make(rng.choice(tasks), rng, split) for _ in range(n)]

    def eval_set(self, n_per_task, tasks=VERIFIABLE_TASKS, seed=1234):
        """ a fixed, balanced set of held-out examples for measuring accuracy """
        rng = random.Random(seed)
        return [self.make(task, rng, 'eval') for task in tasks for _ in range(n_per_task)]

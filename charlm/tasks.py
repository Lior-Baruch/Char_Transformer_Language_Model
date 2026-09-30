"""Synthetic instruction-following tasks built from the pretraining corpus.

Every verifiable task has a single correct answer, so replies can be scored automatically. That gives
labeled data for SFT, correct/incorrect reply pairs for DPO and a reward for GRPO. The "speak" task asks for
a line in the style of a character from the corpus; it has no single right answer and is only used for SFT.

The math tasks (add, sub, mul, div, word) also come with a step-by-step reasoning trace (see reasoning.py) that
SFT can teach the model to write before its answer.

Words, numbers and speeches are split into a train part and a held-out eval part, so evaluation measures whether
the model learned the task rather than memorized the training examples. Word problems also hold out one phrasing
per operation, so their eval score measures whether the model understands wording it never saw.
"""
import random
import re
import zlib
from dataclasses import asdict, dataclass

from .reasoning import TRACES, add_trace, answer_of, div_trace, mul_trace, sub_trace

VERIFIABLE_TASKS = ('reverse', 'uppercase', 'spell', 'length', 'add')  # the original tasks (and the defaults)
ALL_TASKS = VERIFIABLE_TASKS + ('speak',)
NEW_TASKS = ('sub', 'mul', 'div', 'word')  # opt-in: list them in a config's "tasks"
MATH_TASKS = ('add', 'sub', 'mul', 'div', 'word')  # the tasks with a reasoning trace
CHECKABLE_TASKS = VERIFIABLE_TASKS + NEW_TASKS  # every task with a single right answer
TASKS = CHECKABLE_TASKS + ('speak',)  # every task

# word problems: one operation each; a is the first number in the sentence, b the second (for * the number of
# groups, for / the number of friends or boxes). The last phrasing of each operation is only used for evaluation.
WORD_TEMPLATES = {
    '+': ["{n} has {a} {i} and gets {b} more. How many now?",
          "{n} had {a} {i} and found {b} more. How many in total?",
          "{n} has {a} {i}. A friend gives {b} more. How many now?",
          "There are {a} {i} and {b} more arrive. How many in all?"],
    '-': ["{n} has {a} {i} and gives away {b}. How many left?",
          "{n} had {a} {i} and lost {b}. How many are left?",
          "{n} has {a} {i} and uses {b}. How many remain?",
          "There were {a} {i}. {b} were taken. How many remain?"],
    '*': ["{n} buys {b} bags of {a} {i}. How many {i}?",
          "{n} has {b} boxes with {a} {i} in each. How many {i}?",
          "Each of {b} friends has {a} {i}. How many {i} in total?",
          "{n} packs {a} {i} into each of {b} boxes. How many {i}?"],
    '/': ["{n} shares {a} {i} equally among {b} friends. How many each?",
          "{n} has {a} {i} to divide equally among {b} friends. How many each?",
          "{a} {i} are split equally into {b} boxes. How many per box?",
          "{n} puts {a} {i} into {b} equal groups. How many per group?"],
}
NAMES = ['Adam', 'Maya', 'Tom', 'Sara', 'Leo', 'Noa', 'Ben', 'Lily', 'Dana', 'Omer', 'Yael', 'Ron']
ITEMS = ['apples', 'cards', 'pencils', 'marbles', 'cookies', 'stickers', 'books', 'coins', 'shells', 'candies']


@dataclass
class Example:
    task: str
    prompt: str
    answer: str
    reasoning: str = ''  # the step-by-step trace for math tasks ('' for the others)

    def to_dict(self):
        return asdict(self)


def score(example, reply):
    """ reward for a reply: 1.0 if its final answer (the text after <|/think|> when it reasons) is exactly the
    expected answer, ignoring surrounding whitespace; else 0.0 """
    return 1.0 if answer_of(reply).strip() == example.answer else 0.0


def eval_tasks(tasks):
    """ the tasks a training stage evaluates: the original five, plus any new tasks it trains on """
    return VERIFIABLE_TASKS + tuple(t for t in NEW_TASKS if t in tasks)


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

    def __init__(self, text, split_seed=0, eval_fraction=0.2, min_word_len=3, max_word_len=12, max_number=99,
                 max_operand=999):
        self.split_seed, self.eval_fraction, self.max_operand = split_seed, eval_fraction, max_operand
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

    def _held_out(self, name, item):
        """ whether a number problem belongs to the eval split: a stable hash, so the split is the same in every
        run without listing every possible problem """
        return zlib.crc32(f'{self.split_seed}:{name}:{item}'.encode()) % 10_000 < self.eval_fraction * 10_000

    def _draw(self, rng, split, name, draw):
        """ draw problems until one falls in the requested split """
        while True:
            item = draw()
            if self._held_out(name, item) == (split == 'eval'):
                return item

    def _choice(self, rng, pool, split, task):
        items = self.pools[pool][split]
        if not items:
            raise ValueError(f"the corpus has no {pool} for the {task!r} task (the 'speak' task needs a "
                             f"Shakespeare-style corpus such as data/input.txt)")
        return rng.choice(items)

    def make(self, task, rng, split='train'):
        """ one random example of the given task """
        word = lambda: self._choice(rng, 'words', split, task)
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
            trace, answer = add_trace(a, b)
            return Example(task, f"What is {a} + {b}?", answer, trace)
        if task == 'sub':
            def draw():
                a = rng.randint(0, self.max_operand)
                return a, rng.randint(0, a)
            a, b = self._draw(rng, split, 'sub', draw)
            trace, answer = sub_trace(a, b)
            return Example(task, f"What is {a} - {b}?", answer, trace)
        if task == 'mul':
            a, b = self._draw(rng, split, 'mul', lambda: (rng.randint(2, self.max_operand), rng.randint(2, 9)))
            trace, answer = mul_trace(a, b)
            return Example(task, f"What is {a} * {b}?", answer, trace)
        if task == 'div':
            b, q = self._draw(rng, split, 'div', lambda: (rng.randint(2, 9), rng.randint(2, self.max_operand)))
            trace, answer = div_trace(b * q, b)
            return Example(task, f"What is {b * q} / {b}?", answer, trace)
        if task == 'word':
            return self._word_problem(rng, split)
        if task == 'speak':
            speaker, line = self._choice(rng, 'speeches', split, task)
            return Example(task, f"Say a line as {speaker}.", line)
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")

    def _word_problem(self, rng, split):
        op = rng.choice('+-*/')
        templates = WORD_TEMPLATES[op]
        template = templates[-1] if split == 'eval' else rng.choice(templates[:-1])

        def draw():  # small numbers keep the prompt, the trace and the answer within a 128-character context
            if op == '+':
                return rng.randint(2, 50), rng.randint(2, 50)
            if op == '-':
                a = rng.randint(10, 99)
                return a, rng.randint(1, a - 1)
            if op == '*':
                return rng.randint(2, 11), rng.randint(2, 9)
            b = rng.randint(2, 9)
            return b * rng.randint(2, 11), b
        a, b = self._draw(rng, split, f'word{op}', draw)
        trace, answer = TRACES[op](a, b)
        prompt = template.format(n=rng.choice(NAMES), i=rng.choice(ITEMS), a=a, b=b)
        return Example('word', prompt, answer, f'{a}{op}{b}: {trace}')

    def sample(self, n, tasks=ALL_TASKS, split='train', seed=0):
        """ n examples, each of a task picked uniformly at random """
        rng = random.Random(seed)
        return [self.make(rng.choice(tasks), rng, split) for _ in range(n)]

    def eval_set(self, n_per_task, tasks=VERIFIABLE_TASKS, seed=1234):
        """ a fixed, balanced set of held-out examples for measuring accuracy """
        rng = random.Random(seed)
        return [self.make(task, rng, 'eval') for task in tasks for _ in range(n_per_task)]

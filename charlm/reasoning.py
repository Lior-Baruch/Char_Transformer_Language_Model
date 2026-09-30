"""Step-by-step reasoning: the reply format, and the arithmetic scratchpads the math tasks teach.

A reasoning reply writes a scratchpad between <|think|> and <|/think|>, then the final answer:

    <|think|>7+5+0=12 A=2, 4+8+1=13 A=132 => 132<|/think|>132

The scratchpads work digit by digit, like arithmetic on paper, so each step is a small lookup a tiny model can
learn: every step has the same shape, carries (+c) and borrows (bN) are always written, and A= is the answer built
so far. The trace ends with "=> answer", which the model then copies after <|/think|>. Only that final answer is
scored, so a model is free to reason differently, but the traces are what SFT teaches it.

    add  47 + 85  -> 7+5+0=12 A=2, 4+8+1=13 A=132 => 132             (digits from the right, +carry)
    sub  82 - 47  -> 2-7-0=5 b1 A=5, 8-4-1=3 b0 A=35 => 35           (top-bottom-borrow_in = digit, borrow_out)
    mul  47 * 6   -> 7*6=42+0=42 A=2, 4*6=24+4=28 A=282 => 282       (digit*b = product, +carry)
    div  84 / 6   -> 08/6=1 r2 A=1, 24/6=4 r0 A=14 => 14             (long division from the left, r=remainder)
"""
from .tokenizer import END_THINK, THINK


def format_response(answer, reasoning=''):
    """ the text a model should reply with: the answer, preceded by the reasoning when there is some """
    return f"{THINK}{reasoning}{END_THINK}{answer}" if reasoning else answer


def split_reply(reply):
    """ (reasoning, answer) of a reply's text. Without <|think|> the whole reply is the answer; a reply whose
    reasoning was never closed (e.g. cut off by max_new_tokens) has reasoning but no answer """
    if THINK not in reply:
        return '', reply
    after = reply.split(THINK, 1)[1]
    if END_THINK not in after:
        return after, ''
    reasoning, answer = after.split(END_THINK, 1)
    return reasoning, answer


def answer_of(reply):
    """ the final answer in a reply: the text after <|/think|>, or the whole reply when it doesn't reason """
    return split_reply(reply)[1]


def _digits(n, width):
    """ the digits of n from the right (least significant first), zero-padded to width """
    return [int(c) for c in reversed(str(n).zfill(width))]


def _finish(steps, running):
    answer = running.lstrip('0') or '0'
    return ', '.join(steps) + f' => {answer}', answer


def add_trace(a, b):
    """ (trace, answer) for a + b """
    width = max(len(str(a)), len(str(b)))
    steps, running, carry = [], '', 0
    for i, (x, y) in enumerate(zip(_digits(a, width), _digits(b, width))):
        s = x + y + carry
        # the last step writes its whole two-digit sum, so a final carry becomes the leading digit
        running = (f'{s:02d}' if i == width - 1 else str(s % 10)) + running
        steps.append(f'{x}+{y}+{carry}={s:02d} A={running}')
        carry = s // 10
    return _finish(steps, running)


def sub_trace(a, b):
    """ (trace, answer) for a - b, with a >= b """
    assert 0 <= b <= a, "subtraction traces need a >= b >= 0"
    width = len(str(a))
    steps, running, borrow = [], '', 0
    for x, y in zip(_digits(a, width), _digits(b, width)):
        d = x - y - borrow
        out = 1 if d < 0 else 0
        d += 10 * out
        running = str(d) + running
        steps.append(f'{x}-{y}-{borrow}={d} b{out} A={running}')
        borrow = out
    return _finish(steps, running)


def mul_trace(a, b):
    """ (trace, answer) for a * b, with b a single digit """
    assert 0 <= b <= 9, "multiplication traces need a one-digit b"
    width = len(str(a))
    steps, running, carry = [], '', 0
    for i, x in enumerate(_digits(a, width)):
        p = x * b
        s = p + carry
        running = (f'{s:02d}' if i == width - 1 else str(s % 10)) + running
        steps.append(f'{x}*{b}={p:02d}+{carry}={s:02d} A={running}')
        carry = s // 10
    return _finish(steps, running)


def div_trace(a, b):
    """ (trace, answer) for a / b, with b a single digit that divides a exactly """
    assert 1 <= b <= 9 and a % b == 0, "division traces need a one-digit b that divides a"
    steps, running, rem = [], '', 0
    for d in str(a):
        cur = rem * 10 + int(d)
        q, rem = divmod(cur, b)
        running += str(q)
        steps.append(f'{cur:02d}/{b}={q} r{rem} A={running}')
    return _finish(steps, running)


TRACES = {'+': add_trace, '-': sub_trace, '*': mul_trace, '/': div_trace}

"""Character-level tokenizer with a few special tokens for chat formatting."""
import re

# the default special tokens get the first ids; they mark the parts of a chat conversation
PAD = "<|pad|>"  # fills the end of shorter sequences in a batch
USER = "<|user|>"  # starts the user's message
ASSISTANT = "<|assistant|>"  # starts the model's reply
END = "<|end|>"  # ends the model's reply
DEFAULT_SPECIAL_TOKENS = (PAD, USER, ASSISTANT, END)

# reasoning tokens wrap a step-by-step scratchpad before the final answer:
#   <|assistant|><|think|>7+5+0=12 A=2, 4+8+1=13 A=132 => 132<|/think|>132<|end|>
# they are appended to a tokenizer only when a model is taught to reason (see add_special_tokens)
THINK = "<|think|>"
END_THINK = "<|/think|>"
REASONING_TOKENS = (THINK, END_THINK)

# newline plus every printable ASCII character, so any English text, digit or symbol can be encoded
DEFAULT_CHARS = "\n" + "".join(chr(i) for i in range(32, 127))


class CharTokenizer:
    """ maps each character (and each special token) to an integer id and back """

    def __init__(self, chars=DEFAULT_CHARS, special_tokens=DEFAULT_SPECIAL_TOKENS):
        self._build(list(special_tokens) + sorted(set(chars)), special_tokens)

    def _build(self, itos, special_tokens):
        self.itos = list(itos)  # id -> token
        self.special_tokens = list(special_tokens)
        specials = set(self.special_tokens)
        self.chars = [t for t in self.itos if t not in specials]
        self.stoi = {s: i for i, s in enumerate(self.itos)}  # token -> id
        self._special_ids = {self.stoi[t] for t in self.special_tokens}
        self._special_pattern = re.compile('(' + '|'.join(map(re.escape, self.special_tokens)) + ')')

    @classmethod
    def from_text(cls, text, special_tokens=DEFAULT_SPECIAL_TOKENS):
        """ the default characters plus any other character that appears in the text """
        return cls(set(DEFAULT_CHARS) | set(text), special_tokens)

    def add_special_tokens(self, tokens):
        """ append the tokens that are missing after every existing id, so existing ids never change
        (the model's embedding must then grow too: CharTransformerLanguageModel.resize_vocab).
        returns how many tokens were added """
        new = [t for t in tokens if t not in self.stoi]
        if new:
            self._build(self.itos + new, self.special_tokens + new)
        return len(new)

    @property
    def vocab_size(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def user_id(self):
        return self.stoi[USER]

    @property
    def assistant_id(self):
        return self.stoi[ASSISTANT]

    @property
    def end_id(self):
        return self.stoi[END]

    @property
    def has_reasoning_tokens(self):
        return all(t in self.stoi for t in REASONING_TOKENS)

    def encode(self, text, allow_special=True):
        """ text -> list of ids. allow_special says which markers like <|end|> become their special token id:
        True (all of this tokenizer's special tokens), False (none: markers are encoded as plain characters)
        or a collection of special token strings """
        if allow_special is True:
            allowed = set(self.special_tokens)
        elif allow_special is False:
            allowed = set()
        else:
            allowed = set(allow_special) & set(self.special_tokens)
        pieces = self._special_pattern.split(text) if allowed else [text]
        ids = []
        for piece in pieces:
            if piece in allowed:
                ids.append(self.stoi[piece])
                continue
            for c in piece:
                if c not in self.stoi:
                    raise ValueError(f"character {c!r} is not in the tokenizer's vocabulary")
                ids.append(self.stoi[c])
        return ids

    def decode(self, ids, skip_special=False):
        """ list of ids -> text """
        return ''.join(self.itos[i] for i in map(int, ids) if not (skip_special and i in self._special_ids))

    def to_dict(self):
        n = len(self.special_tokens)
        if self.itos[:n] == self.special_tokens and self.itos[n:] == sorted(self.chars):
            return {'chars': ''.join(self.chars), 'special_tokens': self.special_tokens}  # the original layout
        return {'itos': self.itos, 'special_tokens': self.special_tokens}  # special tokens were appended later

    @classmethod
    def from_dict(cls, d):
        tokenizer = cls.__new__(cls)
        if 'itos' in d:
            tokenizer._build(d['itos'], d['special_tokens'])
        else:
            tokenizer._build(list(d['special_tokens']) + sorted(set(d['chars'])), d['special_tokens'])
        return tokenizer

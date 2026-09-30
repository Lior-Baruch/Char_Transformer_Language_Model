"""Character-level tokenizer with a few special tokens for chat formatting."""
import re

# the special tokens get the first ids; they mark the parts of a chat conversation
PAD = "<|pad|>"  # fills the end of shorter sequences in a batch
USER = "<|user|>"  # starts the user's message
ASSISTANT = "<|assistant|>"  # starts the model's reply
END = "<|end|>"  # ends the model's reply
DEFAULT_SPECIAL_TOKENS = (PAD, USER, ASSISTANT, END)

# newline plus every printable ASCII character, so any English text, digit or symbol can be encoded
DEFAULT_CHARS = "\n" + "".join(chr(i) for i in range(32, 127))


class CharTokenizer:
    """ maps each character (and each special token) to an integer id and back """

    def __init__(self, chars=DEFAULT_CHARS, special_tokens=DEFAULT_SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.chars = sorted(set(chars))
        self.itos = self.special_tokens + self.chars  # id -> token
        self.stoi = {s: i for i, s in enumerate(self.itos)}  # token -> id
        self._special_ids = set(range(len(self.special_tokens)))
        self._special_pattern = re.compile('(' + '|'.join(map(re.escape, self.special_tokens)) + ')')

    @classmethod
    def from_text(cls, text, special_tokens=DEFAULT_SPECIAL_TOKENS):
        """ the default characters plus any other character that appears in the text """
        return cls(set(DEFAULT_CHARS) | set(text), special_tokens)

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

    def encode(self, text, allow_special=True):
        """ text -> list of ids; with allow_special, markers like <|end|> become their special token id """
        pieces = self._special_pattern.split(text) if allow_special else [text]
        ids = []
        for piece in pieces:
            if allow_special and piece in self.stoi and len(piece) > 1:
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
        return {'chars': ''.join(self.chars), 'special_tokens': self.special_tokens}

    @classmethod
    def from_dict(cls, d):
        return cls(d['chars'], d['special_tokens'])

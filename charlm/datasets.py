"""Downloading and cleaning larger text corpora for pretraining (python -m charlm prepare-data --help).

    tinystories       ~2.7M short stories in simple English, written by GPT-4 (TinyStoriesV2, ~2.2 GB).
                      A small model learns to write coherent stories from it, far beyond what 1 MB allows
    shakespeare       the complete works of Shakespeare from Project Gutenberg (~5.4 MB, 5x data/input.txt)
    tiny_shakespeare  the 1 MB excerpt that data/input.txt holds, the default corpus
    files             your own .txt files (--files), joined into one

Text is streamed (never held in memory whole) and normalized to the default tokenizer's characters: printable
ASCII plus newline. Curly quotes become straight ones, accented letters lose their accents, "1/2" replaces the
one-character fraction and other characters are dropped. A .json file next to the output records what was
prepared, so running the same command again skips the work. Blank lines and line breaks are kept as they are.
"""
import io
import json
import os
import re
import time
import unicodedata
import urllib.request

SOURCES = {
    'tinystories': {
        'url': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
        'out': 'data/tinystories.txt'},
    'shakespeare': {'url': 'https://www.gutenberg.org/cache/epub/100/pg100.txt', 'out': 'data/shakespeare.txt',
                    'gutenberg': True},
    'tiny_shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'out': 'data/tiny_shakespeare.txt'},
}

# applied before the Unicode decomposition, which would otherwise drop these or turn them into the wrong thing
# (NFKD makes the one-character 1/2 into 1, a fraction slash that is not ASCII, and 2, so it would read "12")
REPLACEMENTS = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'", '\u2032': "'", '\u00b4': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"', '\u2033': '"', '\u00ab': '"', '\u00bb': '"',
    '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-',
    '\u2026': '...', '\u00a0': ' ', '\u2009': ' ', '\u200a': ' ', '\u202f': ' ',
    '\u00bd': '1/2', '\u00bc': '1/4', '\u00be': '3/4', '\u2153': '1/3', '\u2154': '2/3', '\u00d7': 'x',
    '\u00e6': 'ae', '\u00c6': 'AE', '\u0153': 'oe', '\u0152': 'OE', '\u00df': 'ss', '\u00f8': 'o', '\u00d8': 'O',
    '\u0142': 'l', '\u0141': 'L', '\u00f0': 'd', '\u00d0': 'D', '\u00fe': 'th', '\u00de': 'Th', '\u2022': '*',
    '\ufeff': None, '\u200b': None,
})
# control characters: tabs become spaces, the others (except the newline) are dropped
CONTROL = str.maketrans({**{chr(i): None for i in [*range(9), *range(11, 32), 127]}, '\t': ' '})
END_OF_TEXT = '<|endoftext|>'  # TinyStories separates stories with this line
FRACTION_SLASH = '\u2044'  # what NFKD puts between the digits of a fraction character


def normalize(text):
    """ text -> printable ASCII plus newlines (see the module docstring) """
    if not text.isascii():
        text = unicodedata.normalize('NFKD', text.translate(REPLACEMENTS)).replace(FRACTION_SLASH, '/')
        text = text.encode('ascii', 'ignore').decode('ascii')
    return text.translate(CONTROL)


class _Writer:
    """ writes text up to max_chars characters, skipping newlines at the very start """

    def __init__(self, f, max_chars=None):
        self.f, self.max_chars, self.chars, self.tail = f, max_chars, 0, ''

    @property
    def full(self):
        return self.max_chars is not None and self.chars >= self.max_chars

    def write(self, text):
        if not self.chars:
            text = text.lstrip('\n')
        if self.max_chars is not None and self.chars + len(text) > self.max_chars:
            room = self.max_chars - self.chars
            cut = text.rfind('\n', 0, room)
            text = text[:cut + 1] if cut >= 0 else text[:room]  # stop at a line end when there is one
            self.max_chars = self.chars + len(text)  # nothing more fits
        if text:
            self.f.write(text)
            self.chars += len(text)
            self.tail = (self.tail + text)[-2:]

    def blank_line(self):
        """ end the text so far with a blank line (between two files) """
        if self.chars:
            self.write('\n' * (2 - len(self.tail) + len(self.tail.rstrip('\n'))))


def _blocks(stream, block_chars=1 << 23):
    """ successive blocks of whole lines, ~block_chars each """
    while True:
        lines = stream.readlines(block_chars)
        if not lines:
            return
        yield ''.join(lines)


def _clean_blocks(stream, gutenberg=False):
    """ normalized text blocks; for a Project Gutenberg book, only the text between its START and END markers """
    started = not gutenberg
    for block in _blocks(stream):
        if gutenberg:
            if not started:
                match = re.search(r'^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$', block, re.M)
                if not match:
                    continue
                block, started = block[match.end():], True
            match = re.search(r'^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG', block, re.M)
            if match:
                yield normalize(block[:match.start()])
                return
        if END_OF_TEXT in block:  # the separator line becomes a blank line between stories
            block = re.sub(r'^' + re.escape(END_OF_TEXT) + r'\n', '\n', block, flags=re.M)
            block = block.replace(END_OF_TEXT, '\n\n')
        yield normalize(block)
    if not started:
        raise ValueError("no '*** START OF THE PROJECT GUTENBERG' line found; is this a Gutenberg book?")


def _open_url(url, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': 'charlm-prepare-data'})
    return urllib.request.urlopen(request, timeout=timeout)


def _read_url(url, writer, gutenberg):
    """ stream a download into writer; raises if the connection ended before the whole file arrived """
    with _open_url(url) as response:
        stream = io.TextIOWrapper(response, encoding='utf-8', errors='replace')
        _copy(stream, writer, gutenberg)
        if not writer.full:
            stream.read()  # the rest (e.g. a Gutenberg license), so the size check below sees the whole download
            missing = getattr(response, 'length', None)  # bytes the server announced but never sent
            if missing:
                raise IOError(f"the download of {url} ended early ({missing:,} bytes missing); try again")


def _copy(stream, writer, gutenberg=False):
    start = time.time()
    for block in _clean_blocks(stream, gutenberg):
        before = writer.chars
        writer.write(block)
        if writer.chars // 100_000_000 > before // 100_000_000:
            print(f"  {writer.chars / 1e6:,.0f}M characters | {time.time() - start:.0f}s", flush=True)
        if writer.full:
            return


def prepare(source, out=None, files=(), url=None, max_chars=None, force=False):
    """ download (or read) a corpus, normalize it and write it to out; returns the path written.
    source: one of SOURCES or 'files'. url overrides the source's download address. max_chars stops early,
    e.g. to try a pipeline on part of TinyStories """
    if source == 'files':
        if not files or not out:
            raise ValueError("the 'files' source needs files to join and an output path (--files and --out)")
        if url:
            raise ValueError("the 'files' source reads local files; it takes no url")
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise ValueError(f"not a file: {', '.join(missing)}")
    elif source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {', '.join(SOURCES)} or files")
    elif files:
        raise ValueError(f"--files is only for the 'files' source, not {source!r}")
    spec = SOURCES.get(source, {})
    out = out or spec['out']
    url = url or spec.get('url')
    meta_path = os.path.splitext(out)[0] + '.json'
    meta = {'source': source, 'url': url, 'max_chars': max_chars,
            # for local files, their sizes and modification times, so editing one prepares the output again
            'files': [[path, os.path.getsize(path), int(os.path.getmtime(path))] for path in files]}
    if not force and os.path.exists(out) and os.path.exists(meta_path):
        with open(meta_path) as f:
            saved = json.load(f)
        if {k: saved.get(k) for k in meta} == meta and saved.get('chars') == os.path.getsize(out):
            print(f"{out} is already prepared ({saved['chars']:,} characters); skipping (--force redoes it)")
            return out

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp, start = out + '.tmp', time.time()
    try:
        with open(tmp, 'w', encoding='ascii', newline='\n') as f:
            writer = _Writer(f, max_chars)
            if source != 'files':
                print(f"reading {url}")
                _read_url(url, writer, spec.get('gutenberg', False))
            for path in files:
                print(f"reading {path}")
                writer.blank_line()
                with open(path, encoding='utf-8', errors='replace') as stream:
                    _copy(stream, writer)
                if writer.full:
                    break
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)  # a partial file is never left behind, or mistaken for a finished one
        raise
    with open(meta_path, 'w') as f:
        json.dump({**meta, 'chars': os.path.getsize(out)}, f, indent=2)
    print(f"wrote {os.path.getsize(out):,} characters to {out} in {time.time() - start:.0f}s")
    return out

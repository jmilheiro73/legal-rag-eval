"""Statute corpus: parsing and chunking.

Corpus files are Markdown with a small front matter (code, title, act, citation, url, retrieved) followed by
articles. An article starts at a line beginning with "Art."; each provision (caput, inciso, parágrafo) is on its own
line; "[...]" marks provisions left out of the excerpt.

Chunking follows the pilot's rule for statutes: one chunk per article, prefixed with the statute ("[CDC, Lei
8.078/1990] Art. 26. ..."), so the citation travels with the text. An article longer than the token budget is split
by provision, and each piece carries the caput (and the paragraph it belongs to) so it still reads on its own; its
vector is computed from the short form "[LGPD, ...] Art. 7º, IX - ..." so that a long caput shared by every piece
does not dominate the embedding of each provision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CHUNKER_VERSION = "statute-1"
OMITTED = "[...]"
_ARTICLE_START = re.compile(r"^Art\.\s*(?P<num>\d{1,3}(?:\.\d{3})+|\d+)[º°o]?(?:-(?P<suffix>[A-Z]))?(?=[\s.\-])")
_PARAGRAPH = re.compile(r"^(?:§\s*\d+|Parágrafo único)")
_INCISO = re.compile(r"^[IVXLC]+\s*[-–—]")
_FRONT_MATTER_KEYS = ("code", "title", "act", "citation", "url", "retrieved")


def estimate_tokens(text: str) -> int:
    """About 3.5 characters per token in Portuguese; good enough for a budget, no tokenizer dependency."""
    return max(1, round(len(text) / 3.5))


@dataclass(frozen=True)
class Law:
    code: str  # "CDC", "Lei 12.506": the prefix of every article id
    title: str
    act: str
    citation: str  # "Lei 8.078/1990"
    url: str
    retrieved: str

    @property
    def prefix(self) -> str:
        return f"[{self.citation}]" if self.citation.startswith(self.code) else f"[{self.code}, {self.citation}]"


@dataclass(frozen=True)
class Article:
    law: Law
    number: str  # "26", "476-A", "1.009"
    lines: tuple[str, ...]  # first line is the caput, starting with "Art."

    @property
    def id(self) -> str:
        return f"{self.law.code} art. {self.number}"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_id: str
    law: str
    text: str  # what the model reads
    embed_text: str  # what is embedded

    @property
    def source(self) -> str:
        return self.article_id


def _front_matter(raw: str, path: Path) -> tuple[Law, str]:
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    missing = [k for k in _FRONT_MATTER_KEYS if not meta.get(k)]
    if missing:
        raise ValueError(f"{path}: front matter without {missing}")
    return Law(**{k: meta[k] for k in _FRONT_MATTER_KEYS}), raw[m.end() :]


def parse_corpus_file(path: str | Path) -> list[Article]:
    path = Path(path)
    law, body = _front_matter(path.read_text(encoding="utf-8"), path)
    articles: list[Article] = []
    number, lines = "", list[str]()
    for line in (raw.strip() for raw in body.splitlines()):
        if not line:
            continue
        m = _ARTICLE_START.match(line)
        if m:
            if lines:
                articles.append(Article(law, number, tuple(lines)))
            number = m.group("num") + (f"-{m.group('suffix')}" if m.group("suffix") else "")
            lines = [line]
        elif lines:
            lines.append(line)
        else:
            raise ValueError(f"{path}: text before the first article: {line[:60]!r}")
    if lines:
        articles.append(Article(law, number, tuple(lines)))
    return articles


def load_corpus(directory: str | Path) -> list[Article]:
    files = sorted(Path(directory).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no corpus files (*.md) in {directory}")
    articles = [a for f in files for a in parse_corpus_file(f)]
    ids = [a.id for a in articles]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"duplicated article ids in the corpus: {duplicated}")
    return articles


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " (...)"


def chunk_article(article: Article, max_tokens: int = 400) -> list[Chunk]:
    prefix = article.law.prefix
    whole = f"{prefix} {article.text}"
    if estimate_tokens(whole) <= max_tokens or len(article.lines) == 1:
        return [Chunk(f"{article.id}#0", article.id, article.law.code, whole, whole)]
    caput = article.lines[0]
    head = _ARTICLE_START.match(caput)
    label = caput[: head.end()] if head else f"Art. {article.number}"
    chunks: list[Chunk] = []
    paragraph = ""
    for line in article.lines[1:]:
        if line == OMITTED:
            continue
        if _PARAGRAPH.match(line):
            paragraph = line
            context = [f"{prefix} {_short(caput, 400)}"]
        elif _INCISO.match(line) and paragraph:
            context = [f"{prefix} {_short(caput, 400)}", _short(paragraph, 200)]
        else:
            context = [f"{prefix} {_short(caput, 400)}"]
        text = "\n".join([*context, line])
        chunks.append(
            Chunk(f"{article.id}#{len(chunks)}", article.id, article.law.code, text, f"{prefix} {label}, {line}")
        )
    return chunks or [Chunk(f"{article.id}#0", article.id, article.law.code, whole, whole)]


def chunk_corpus(articles: list[Article], max_tokens: int = 400) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, max_tokens)]

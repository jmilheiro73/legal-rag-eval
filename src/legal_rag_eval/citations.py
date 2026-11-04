"""Rule-based source check: no model call.

Extracts the articles, súmulas and statutes cited in an answer and checks that each one appears in the passages
that were retrieved for it. A small model sometimes cites from memory; a citation that is not in any retrieved
passage is reported as unsupported. The same extractor is used by the legal-basis check in ``checks.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

_NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)"
_PATTERNS: dict[str, re.Pattern[str]] = {
    # "art. 476-A", "artigo 476A", "arts. 186". The suffix letter must be uppercase: in "arts. 1.029 e 105",
    # "e" is not a suffix.
    "art": re.compile(r"\bart(?:igo)?s?\.?\s*" + _NUM + r"(?:(?:\s*-\s*)?(?-i:([A-Z]))\b)?", re.I),
    "sumula": re.compile(r"\bs[uú]mulas?(?:\s+vinculantes?)?\s*(?:n[º°o.]*\s*)?" + _NUM, re.I),
    "lei": re.compile(r"\blei(?:\s+complementar)?\s*(?:n[º°o.]*\s*)?" + _NUM + r"(?:/\d{2,4})?", re.I),
}
_LABEL = {"art": "art.", "sumula": "Súmula", "lei": "Lei"}


def normalize_number(raw: str, suffix: str | None = None) -> str:
    """'1.009' -> '1009', ('476', 'A') -> '476a'. Thousands separators and suffix case do not matter."""
    return raw.replace(".", "") + (suffix or "").lower()


@dataclass(frozen=True)
class Citation:
    kind: str  # art | sumula | lei
    number: str  # normalised: "1009", "476a"
    text: str  # as displayed: "art. 1.009", "art. 476-A"

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.number


def extract(text: str) -> list[Citation]:
    """Citations in order of appearance, without repeats ("artigo 476-A" and "art. 476A" are the same one)."""
    found: list[tuple[int, Citation]] = []
    for kind, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            suffix = m.group(2) if kind == "art" else None
            label = f"{_LABEL[kind]} {m.group(1)}" + (f"-{suffix.upper()}" if suffix else "")
            found.append((m.start(), Citation(kind, normalize_number(m.group(1), suffix), label)))
    seen: dict[tuple[str, str], Citation] = {}
    for _, c in sorted(found, key=lambda x: x[0]):
        seen.setdefault(c.key, c)
    return list(seen.values())


class SourcedText(Protocol):
    """Anything with a text and a label: a retrieved passage, in practice."""

    @property
    def text(self) -> str: ...

    @property
    def source(self) -> str: ...


@dataclass
class SourceCheck:
    cited: list[Citation] = field(default_factory=list)
    unsupported: list[Citation] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # passages that contain at least one of the cited items

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "cited": [c.text for c in self.cited],
            "unsupported": [c.text for c in self.unsupported],
            "sources": self.sources,
        }


def check_sources(answer: str, passages: Iterable[SourcedText]) -> SourceCheck:
    """Every citation in the answer must appear in at least one passage. Matching is by kind and number only:
    "art. 7" in a CF passage also supports an "art. 7" meant as the LGPD. That is a known limit of the rule."""
    cited = extract(answer)
    if not cited:
        return SourceCheck()
    per_passage = [(p, {c.key for c in extract(p.text)}) for p in passages]
    unsupported: list[Citation] = []
    sources: list[str] = []
    for c in cited:
        matches = [p for p, keys in per_passage if c.key in keys]
        if not matches:
            unsupported.append(c)
        for p in matches:
            if p.source not in sources:
                sources.append(p.source)
    return SourceCheck(cited, unsupported, sources)

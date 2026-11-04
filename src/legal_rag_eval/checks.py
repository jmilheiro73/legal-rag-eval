"""Rule-based answer checks. Deterministic, free to run, and identical between runs, so two runs can be compared
question by question.

A plain keyword match passes wrong answers: a small model often repeats the question back, so a key term from the
question shows up in a wrong answer. Hence four kinds of rule:

- expected terms: every entry must appear; "a|b" means either. Terms that start or end with a digit match whole
  numbers only, so "15" is not found in "1.015", "150" or "15,5", and "2%" is not found in "12%".
- yes/no: the expected word ("sim" or "não") must be among the first six words, and the answer must not start with
  the opposite one. "Sim, os prazos não contam nos feriados" contains "não" and is still a "sim".
- forbidden terms: any match fails the answer (typically a rule from repealed law the model remembers).
- legal basis: the answer must cite at least one of the expected articles, found with the citation extractor
  ("art. 1.009" and "artigo 1009" are the same). A statute named by number can also be cited as a whole ("Lei 12.506").

Only the text before a verification note (``SEPARATOR``) is scored, because a backend may append its own list of
sources, and those would otherwise satisfy the legal-basis rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .answer_key import AnswerKeyItem, ArticleRef
from .citations import SourcedText, check_sources, extract

SEPARATOR = "\n\n---\n"
YES_NO_WINDOW = 6
_OPPOSITE = {"sim": "não", "não": "sim"}
_WORD = re.compile(r"[a-zà-öø-ÿ]+")

# Failure kinds, in the order the report lists them.
MISSING_NUMBER = "missing_number"
MISSING_TERM = "missing_term"
YES_NO = "yes_no"
FORBIDDEN = "forbidden"
LEGAL_BASIS = "legal_basis"
FAILURE_KINDS = (MISSING_NUMBER, MISSING_TERM, YES_NO, FORBIDDEN, LEGAL_BASIS)


def model_text(answer: str) -> str:
    return answer.split(SEPARATOR, 1)[0]


def first_words(text: str, n: int = YES_NO_WINDOW) -> list[str]:
    return _WORD.findall(text.lower())[:n]


def term_present(term: str, text_lower: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    pattern = re.escape(term)
    if term[0].isdigit():
        pattern = r"(?<![\d.,])" + pattern
    if term[-1].isdigit():
        pattern += r"(?!\d|\.\d|,0*[1-9])"  # "15" not in "15.000" or "15,5"; "50.000.000" is in "50.000.000,00"
    return re.search(pattern, text_lower) is not None


def matching_alternative(spec: str, text_lower: str) -> str | None:
    """First alternative of "a|b|c" found in the text, or None."""
    return next((alt for alt in spec.split("|") if term_present(alt, text_lower)), None)


def is_numeric_spec(spec: str) -> bool:
    return any(ch.isdigit() for ch in spec)


@dataclass(frozen=True)
class Failure:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def check_expected(text: str, expected: Sequence[str]) -> list[Failure]:
    low = text.lower()
    return [
        Failure(MISSING_NUMBER if is_numeric_spec(spec) else MISSING_TERM, spec)
        for spec in expected
        if matching_alternative(spec, low) is None
    ]


def check_yes_no(text: str, expectation: str | None) -> list[Failure]:
    if not expectation:
        return []
    words = first_words(text)
    if expectation not in words or (words and words[0] == _OPPOSITE[expectation]):
        return [Failure(YES_NO, f"expected '{expectation}' at the start")]
    return []


def check_forbidden(text: str, forbidden: Sequence[str]) -> list[Failure]:
    low = text.lower()
    return [Failure(FORBIDDEN, found) for spec in forbidden if (found := matching_alternative(spec, low))]


def check_legal_basis(text: str, legal_basis: Sequence[str]) -> list[Failure]:
    if not legal_basis:
        return []
    cited = {c.key for c in extract(text)}
    for ref in map(ArticleRef.parse, legal_basis):
        if ("art", ref.normalized) in cited:
            return []
        if ref.statute_number and ("lei", ref.statute_number) in cited:
            return []
    return [Failure(LEGAL_BASIS, " or ".join(legal_basis))]


@dataclass
class Grade:
    passed: bool
    failures: list[Failure] = field(default_factory=list)
    unsupported_citations: list[str] | None = None  # None when no passages were available to check against
    cited: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [asdict(f) for f in self.failures],
            "unsupported_citations": self.unsupported_citations,
            "cited": self.cited,
        }


def grade(item: AnswerKeyItem, answer: str, passages: Sequence[SourcedText] | None = None) -> Grade:
    """Pass/fail from the four rules. With passages, also the source check (reported, not part of pass/fail)."""
    text = model_text(answer)
    failures = (
        check_expected(text, item.expected)
        + check_yes_no(text, item.starts_with)
        + check_forbidden(text, item.forbidden)
        + check_legal_basis(text, item.legal_basis)
    )
    unsupported: list[str] | None = None
    if passages is not None:
        unsupported = [c.text for c in check_sources(text, passages).unsupported]
    return Grade(not failures, failures, unsupported, [c.text for c in extract(text)])

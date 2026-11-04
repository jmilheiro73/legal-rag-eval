"""Answer-key schema and loader.

One JSON list; each item:

    {"id": "cpc-03",
     "question": "A Fazenda Pública tem prazo diferenciado ...?",
     "expected": ["dobro"],              # every entry must appear; "a|b" means a or b; digits match whole numbers
     "starts_with": "sim",               # optional, yes/no questions: "sim" or "não"
     "forbidden": ["quádruplo"],         # optional, any of these makes the answer wrong
     "legal_basis": ["CPC art. 183"],    # article ids from the corpus: retrieval target and required citation
     "note": "..."}                      # optional, for humans

The private pilot uses the same rules with Portuguese keys (pergunta, esperado, inicio, nao_pode, fonte) and no id.
Those keys are accepted too, so a key in that format can be scored with this harness unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .citations import normalize_number

YES_NO = ("sim", "não")
_ALIASES = {
    "pergunta": "question",
    "esperado": "expected",
    "inicio": "starts_with",
    "nao_pode": "forbidden",
    "fonte": "note",
}
_ARTICLE_ID = re.compile(r"^(?P<law>.+?)\s+art\.\s*(?P<num>\d{1,3}(?:\.\d{3})+|\d+)(?:-(?P<suffix>[A-Z]))?$")


@dataclass(frozen=True)
class ArticleRef:
    """'CLT art. 476-A' -> law 'CLT', number '476-A', key ('CLT', '476a')."""

    law: str
    number: str

    @classmethod
    def parse(cls, article_id: str) -> ArticleRef:
        m = _ARTICLE_ID.match(article_id.strip())
        if not m:
            raise ValueError(f"not an article id (expected '<LAW> art. <N>'): {article_id!r}")
        number = m.group("num") + (f"-{m.group('suffix')}" if m.group("suffix") else "")
        return cls(m.group("law"), number)

    @property
    def id(self) -> str:
        return f"{self.law} art. {self.number}"

    @property
    def normalized(self) -> str:
        raw, _, suffix = self.number.partition("-")
        return normalize_number(raw, suffix or None)

    @property
    def statute_number(self) -> str | None:
        """For statutes named by number ('Lei 12.506'), the number a citation of the whole statute would carry."""
        m = re.fullmatch(r"Lei\s+(\d{1,3}(?:\.\d{3})+|\d+)", self.law)
        return normalize_number(m.group(1)) if m else None


@dataclass(frozen=True)
class AnswerKeyItem:
    id: str
    question: str
    expected: list[str] = field(default_factory=list)
    starts_with: str | None = None
    forbidden: list[str] = field(default_factory=list)
    legal_basis: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError(f"{self.id}: empty question")
        if self.starts_with is not None and self.starts_with not in YES_NO:
            raise ValueError(f"{self.id}: starts_with must be one of {YES_NO}, got {self.starts_with!r}")
        for ref in self.legal_basis:
            ArticleRef.parse(ref)
        if not (self.expected or self.starts_with or self.legal_basis):
            raise ValueError(f"{self.id}: nothing to check")

    @property
    def refs(self) -> list[ArticleRef]:
        return [ArticleRef.parse(r) for r in self.legal_basis]

    @property
    def law(self) -> str:
        """Area used to group the summary table: the law of the first legal-basis article."""
        return self.refs[0].law if self.legal_basis else "-"

    def rules(self) -> dict[str, Any]:
        """The checkable part of the item, as stored in a run and as Langfuse expected_output."""
        out: dict[str, Any] = {"expected": self.expected}
        if self.starts_with:
            out["starts_with"] = self.starts_with
        if self.forbidden:
            out["forbidden"] = self.forbidden
        if self.legal_basis:
            out["legal_basis"] = self.legal_basis
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any], default_id: str) -> AnswerKeyItem:
        data = {_ALIASES.get(k, k): v for k, v in raw.items()}
        unknown = set(data) - {"id", "question", "expected", "starts_with", "forbidden", "legal_basis", "note"}
        if unknown:
            raise ValueError(f"{data.get('id', default_id)}: unknown fields {sorted(unknown)}")
        return cls(
            id=str(data.get("id") or default_id),
            question=str(data["question"]),
            expected=list(data.get("expected", [])),
            starts_with=data.get("starts_with"),
            forbidden=list(data.get("forbidden", [])),
            legal_basis=list(data.get("legal_basis", [])),
            note=str(data.get("note", "")),
        )


def load_answer_key(path: str | Path) -> list[AnswerKeyItem]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: the answer key must be a JSON list")
    items = [AnswerKeyItem.from_dict(r, f"{path.stem}-{i:02d}") for i, r in enumerate(raw, 1)]
    ids = [it.id for it in items]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"{path}: duplicated ids {duplicated}")
    return items


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

"""Retrieval-only evaluation: no language model, seconds for the whole key.

For each question, retrieve the top max(k) chunks and measure, for every k:

- hit@k: at least one expected article is among the top k chunks;
- recall@k: share of the expected articles among the top k chunks;
- coverage@k: every expected term of the answer key appears in the text of the top k chunks. This is the pilot's
  measure and an upper bound for end-to-end accuracy: what is not in the context, the model can only get right
  from memory. Items without expected terms fall back to hit@k.

Plus MRR (reciprocal rank of the first expected article within max(k)). Use it to compare retrieval settings before
spending a full run with the model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .answer_key import AnswerKeyItem
from .checks import matching_alternative
from .rag import Retriever
from .store import Passage


@dataclass
class RetrievalResult:
    id: str
    question: str
    gold: list[str]
    retrieved: list[str]  # article id of each retrieved chunk, in rank order (repeats when an article is split)
    ranks: dict[str, int | None]  # gold article -> 1-based rank of its first chunk
    missing_terms: dict[int, list[str]]  # k -> expected terms absent from the top k chunks

    def hit(self, k: int) -> bool:
        return any(r is not None and r <= k for r in self.ranks.values())

    def recall(self, k: int) -> float:
        return sum(r is not None and r <= k for r in self.ranks.values()) / len(self.ranks) if self.ranks else 0.0

    def covered(self, k: int, has_terms: bool) -> bool:
        return not self.missing_terms[k] if has_terms else self.hit(k)

    def reciprocal_rank(self) -> float:
        found = [r for r in self.ranks.values() if r is not None]
        return 1.0 / min(found) if found else 0.0


@dataclass
class RetrievalReport:
    ks: list[int]
    results: list[RetrievalResult]
    has_terms: dict[str, bool]
    seconds: float
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[int, dict[str, float]]:
        n = len(self.results) or 1
        return {
            k: {
                "hit": sum(r.hit(k) for r in self.results) / n,
                "recall": sum(r.recall(k) for r in self.results) / n,
                "coverage": sum(r.covered(k, self.has_terms[r.id]) for r in self.results) / n,
            }
            for k in self.ks
        }

    def counts(self, k: int) -> dict[str, int]:
        return {
            "hit": sum(r.hit(k) for r in self.results),
            "coverage": sum(r.covered(k, self.has_terms[r.id]) for r in self.results),
        }

    def mrr(self) -> float:
        return sum(r.reciprocal_rank() for r in self.results) / (len(self.results) or 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "ks": self.ks,
            "seconds": round(self.seconds, 2),
            "mrr": self.mrr(),
            "summary": {str(k): v for k, v in self.summary().items()},
            "results": [asdict(r) for r in self.results],
        }


def score_passages(item: AnswerKeyItem, passages: Sequence[Passage], ks: Sequence[int]) -> RetrievalResult:
    retrieved = [p.article_id for p in passages]
    ranks: dict[str, int | None] = {
        g: next((i for i, a in enumerate(retrieved, 1) if a == g), None) for g in item.legal_basis
    }
    missing: dict[int, list[str]] = {}
    for k in ks:
        context = "\n".join(p.text for p in passages[:k]).lower()
        missing[k] = [spec for spec in item.expected if matching_alternative(spec, context) is None]
    return RetrievalResult(item.id, item.question, list(item.legal_basis), retrieved, ranks, missing)


def evaluate_retrieval(
    items: Sequence[AnswerKeyItem],
    retriever: Retriever,
    ks: Sequence[int] = (1, 3, 5),
    meta: dict[str, Any] | None = None,
) -> RetrievalReport:
    ks = sorted(set(ks))
    t0 = time.perf_counter()
    results = [score_passages(it, retriever.retrieve(it.question, ks[-1]), ks) for it in items]
    return RetrievalReport(
        list(ks), results, {it.id: bool(it.expected) for it in items}, time.perf_counter() - t0, meta or {}
    )


def render_markdown(report: RetrievalReport) -> str:
    n = len(report.results)
    lines = ["# Retrieval evaluation", ""]
    for key, value in report.meta.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        f"{n} questions, {report.seconds:.1f} s, MRR@{report.ks[-1]} {report.mrr():.3f}",
        "",
        "| k | hit@k | recall@k | coverage@k |",
        "|---|---|---|---|",
    ]
    for k, s in report.summary().items():
        c = report.counts(k)
        lines.append(
            f"| {k} | {c['hit']}/{n} ({s['hit']:.0%}) | {s['recall']:.0%} | {c['coverage']}/{n} ({s['coverage']:.0%}) |"
        )
    k = report.ks[-1]
    misses = [r for r in report.results if not (r.hit(k) and r.covered(k, report.has_terms[r.id]))]
    lines += ["", f"## Misses at k={k}", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        what = []
        if not r.hit(k):
            what.append(f"expected {', '.join(r.gold)} not retrieved")
        if r.missing_terms[k]:
            what.append(f"terms not in context: {', '.join(r.missing_terms[k])}")
        lines.append(
            f"- **{r.id}** {r.question}  \n  {'; '.join(what)}. Retrieved: {', '.join(r.retrieved) or 'nothing'}"
        )
    return "\n".join(lines) + "\n"

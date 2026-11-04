"""End-to-end evaluation: ask every question, grade every answer with the rule-based checks.

When the pipeline exposes its passages, each miss is attributed: "retrieval" when no expected article was retrieved
(no prompt change can fix it), "generation" when it was retrieved and the answer is still wrong. Runs are saved as
JSONL (one meta line, one line per question) so they can be re-scored with a stricter key or compared question by
question without calling the model again.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .answer_key import AnswerKeyItem
from .checks import grade
from .rag import Pipeline, PipelineAnswer

RETRIEVAL, GENERATION = "retrieval", "generation"


@dataclass
class QuestionResult:
    id: str
    question: str
    law: str
    rules: dict[str, Any]  # the answer-key rules this answer was graded with
    answer: str
    seconds: float
    grade: dict[str, Any]
    passages: list[dict[str, Any]] | None = None
    retrieval_hit: bool | None = None
    cause: str | None = None  # why it failed: retrieval | generation | None (passed, or unknown)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return bool(self.grade["passed"])


@dataclass
class Run:
    meta: dict[str, Any]
    results: list[QuestionResult]

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)


def attribute(passed: bool, gold: Sequence[str], retrieved: Sequence[str] | None) -> tuple[bool | None, str | None]:
    if retrieved is None or not gold:
        return None, None
    hit = any(g in retrieved for g in gold)
    if passed:
        return hit, None
    return hit, GENERATION if hit else RETRIEVAL


def result_for(item: AnswerKeyItem, out: PipelineAnswer) -> QuestionResult:
    g = grade(item, out.answer, out.passages)
    retrieved = [p.article_id for p in out.passages] if out.passages is not None else None
    hit, cause = attribute(g.passed, item.legal_basis, retrieved)
    passages = None
    if out.passages is not None:
        passages = [
            {"chunk_id": p.chunk_id, "article_id": p.article_id, "score": round(p.score, 4)} for p in out.passages
        ]
    return QuestionResult(
        item.id,
        item.question,
        item.law,
        item.rules(),
        out.answer,
        round(out.seconds, 2),
        g.as_dict(),
        passages,
        hit,
        cause,
        out.usage,
        out.error,
    )


def evaluate_item(item: AnswerKeyItem, pipeline: Pipeline, traceparent: str | None = None) -> QuestionResult:
    return result_for(item, pipeline.answer(item.question, traceparent=traceparent))


Progress = Callable[[int, int, QuestionResult], None]


def print_progress(i: int, n: int, r: QuestionResult) -> None:
    failures = "; ".join(f"{f['kind']}: {f['detail']}" for f in r.grade["failures"])
    print(
        f"[{i}/{n}] {'OK  ' if r.passed else 'FAIL'} {r.seconds:6.1f}s {r.id}" + (f" | {failures}" if failures else ""),
        flush=True,
    )


def run_local(
    items: Sequence[AnswerKeyItem], pipeline: Pipeline, progress: Progress | None = print_progress
) -> list[QuestionResult]:
    results = []
    for i, item in enumerate(items, 1):
        r = evaluate_item(item, pipeline)
        results.append(r)
        if progress:
            progress(i, len(items), r)
    return results


def save_run(run: Run, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", **run.meta}, ensure_ascii=False) + "\n")
        for r in run.results:
            f.write(json.dumps({"type": "result", **asdict(r)}, ensure_ascii=False) + "\n")
    return path


def load_run(path: str | Path) -> Run:
    meta: dict[str, Any] = {}
    results: list[QuestionResult] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.pop("type", "result")
        if kind == "meta":
            meta = row
        else:
            results.append(QuestionResult(**row))
    return Run(meta, results)

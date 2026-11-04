"""Optional Langfuse integration: the answer key becomes a dataset, each run becomes a dataset experiment with
per-item scores. Without LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (or LANGFUSE_BASE_URL), or
without the ``langfuse`` package, nothing is sent and the run goes on locally.

Each request made inside the experiment carries a W3C ``traceparent`` header for the item's observation, so a backend
that is itself instrumented with Langfuse (the black-box mode) nests its own trace under the experiment item.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .answer_key import AnswerKeyItem
from .checks import grade
from .e2e_eval import Progress, QuestionResult, evaluate_item
from .rag import Pipeline

_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langfuse_settings(env: Mapping[str, str] | None = None) -> dict[str, str] | None:
    env = os.environ if env is None else env
    host = env.get("LANGFUSE_HOST") or env.get("LANGFUSE_BASE_URL")
    if not host or not all(env.get(k) for k in _KEYS):
        return None
    return {"base_url": host, "public_key": env["LANGFUSE_PUBLIC_KEY"], "secret_key": env["LANGFUSE_SECRET_KEY"]}


def traceparent(trace_id: str | None, observation_id: str | None) -> str | None:
    return f"00-{trace_id}-{observation_id}-01" if trace_id and observation_id else None


def run_with_langfuse(
    items: Sequence[AnswerKeyItem],
    pipeline: Pipeline,
    dataset: str,
    run_name: str,
    metadata: dict[str, str],
    progress: Progress | None = None,
    env: Mapping[str, str] | None = None,
) -> list[QuestionResult] | None:
    """Run the key as a Langfuse experiment. Returns None (and does nothing) when Langfuse is not configured."""
    settings = langfuse_settings(env)
    if settings is None:
        return None
    try:
        from langfuse import Evaluation, Langfuse
    except ImportError:
        print(
            "LANGFUSE_* is set but the langfuse package is not installed (pip install '.[langfuse]'); "
            "running without it",
            file=sys.stderr,
        )
        return None

    lf = Langfuse(base_url=settings["base_url"], public_key=settings["public_key"], secret_key=settings["secret_key"])
    with contextlib.suppress(Exception):  # the dataset already exists
        lf.create_dataset(name=dataset, description="legal-rag-eval answer key")
    by_id = {it.id: it for it in items}
    order = {it.id: i for i, it in enumerate(items)}
    for it in items:  # fixed ids: running again updates the items instead of duplicating them
        lf.create_dataset_item(
            dataset_name=dataset,
            id=f"{dataset}:{it.id}",
            input=it.question,
            expected_output=it.rules(),
            metadata={"id": it.id, "law": it.law},
        )
    data = sorted(
        (x for x in lf.get_dataset(dataset).items if (x.metadata or {}).get("id") in by_id),
        key=lambda x: order[(x.metadata or {})["id"]],
    )
    results: list[QuestionResult] = []

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        key_item = by_id[item.metadata["id"]]
        tp = traceparent(lf.get_current_trace_id(), lf.get_current_observation_id())
        r = evaluate_item(key_item, pipeline, traceparent=tp)
        results.append(r)
        if progress:
            progress(len(results), len(data), r)
        return {"id": r.id, "answer": r.answer, "seconds": r.seconds, "passages": r.passages}

    def evaluator(*, output: dict[str, Any], **_: Any) -> list[Any]:
        g = grade(by_id[output["id"]], output["answer"])
        r = next(x for x in results if x.id == output["id"])
        evals = [
            Evaluation(name="passed", value=1.0 if g.passed else 0.0, comment="; ".join(map(str, g.failures)) or None),
            Evaluation(name="seconds", value=float(output["seconds"])),
        ]
        if r.grade.get("unsupported_citations") is not None:
            evals.append(Evaluation(name="unsupported_citations", value=float(len(r.grade["unsupported_citations"]))))
        return evals

    lf.run_experiment(
        name=dataset,
        run_name=run_name,
        data=data,
        task=task,
        evaluators=[evaluator],
        max_concurrency=1,
        metadata=metadata,
    )
    lf.flush()
    print(f"Langfuse: dataset '{dataset}', experiment run '{run_name}'", file=sys.stderr)
    results.sort(key=lambda r: order[r.id])
    return results

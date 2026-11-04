"""Markdown report of an end-to-end run: summary tables first, then every question with its answer."""

from __future__ import annotations

from collections import Counter, defaultdict

from .checks import FAILURE_KINDS
from .e2e_eval import GENERATION, RETRIEVAL, Run

_KIND_LABEL = {
    "missing_number": "Missing number",
    "missing_term": "Missing term",
    "yes_no": "Wrong yes/no",
    "forbidden": "Forbidden term",
    "legal_basis": "No legal basis cited",
}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def render_run(run: Run) -> str:
    results, n = run.results, len(run.results)
    seconds = sum(r.seconds for r in results)
    lines = ["# End-to-end evaluation", ""]
    lines += [f"- {k}: {v}" for k, v in run.meta.items()]
    lines += [
        "",
        f"**Passed: {_pct(run.passed, n)}**, {seconds / (n or 1):.1f} s per question, {seconds / 60:.1f} min in total.",
        "",
    ]

    kinds = Counter(f["kind"] for r in results for f in r.grade["failures"])
    lines += [
        "## Failed checks",
        "",
        "A failed answer can fail more than one check.",
        "",
        "| Check | Answers |",
        "|---|---|",
    ]
    lines += [
        f"| {_KIND_LABEL[k]} | {sum(1 for r in results if any(f['kind'] == k for f in r.grade['failures']))} |"
        for k in FAILURE_KINDS
    ]
    lines.append(f"| (failure instances) | {sum(kinds.values())} |")

    failed = [r for r in results if not r.passed]
    causes = Counter(r.cause for r in failed)
    if any(r.passages is not None for r in results):
        lines += [
            "",
            "## Why the misses happened",
            "",
            "| Cause | Misses |",
            "|---|---|",
            f"| Retrieval (no expected article in the context) | {causes.get(RETRIEVAL, 0)} |",
            f"| Generation (expected article retrieved, answer wrong) | {causes.get(GENERATION, 0)} |",
        ]
        unknown = len(failed) - causes.get(RETRIEVAL, 0) - causes.get(GENERATION, 0)
        if unknown:
            lines.append(f"| Unknown (no legal basis in the key) | {unknown} |")
        with_unsupported = [r for r in results if r.grade.get("unsupported_citations")]
        lines += [
            "",
            f"Source check: {len(with_unsupported)} of {n} answers cite something that is not in the retrieved "
            "passages (not counted as a failure; listed per question below).",
        ]

    by_law: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_law[r.law].append(r.passed)
    lines += ["", "## By law", "", "| Law | Passed |", "|---|---|"]
    lines += [f"| {law} | {_pct(sum(v), len(v))} |" for law, v in sorted(by_law.items())]

    lines += ["", "## Questions", ""]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.id}: {r.question}")
        lines.append("")
        detail = [f"**{status}**", f"{r.seconds:.1f} s"]
        if r.usage.get("prompt_tokens"):
            detail.append(f"prompt {r.usage['prompt_tokens']} tokens")
        if r.cause:
            detail.append(f"cause: {r.cause}")
        lines.append(", ".join(detail))
        lines.append("")
        for f in r.grade["failures"]:
            lines.append(f"- {_KIND_LABEL[f['kind']]}: {f['detail']}")
        if r.passages is not None:
            retrieved = ", ".join(dict.fromkeys(p["article_id"] for p in r.passages)) or "nothing"
            lines.append(f"- Expected: {', '.join(r.rules.get('legal_basis', [])) or '-'}; retrieved: {retrieved}")
        if r.grade.get("unsupported_citations"):
            lines.append(f"- Unsupported citations: {', '.join(r.grade['unsupported_citations'])}")
        lines += ["", "> " + r.answer.strip().replace("\n", "\n> "), ""]
    return "\n".join(lines) + "\n"

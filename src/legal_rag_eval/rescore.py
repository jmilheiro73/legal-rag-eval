"""Re-score saved runs and compare runs, without calling the model.

- ``rescore``: grade the stored answers again with the current answer key. When the key gets stricter, old runs
  must be re-scored with it or the comparison between runs is not fair.
- ``compare``: question-by-question transitions between two runs. A change is accepted when the total goes up and
  no question goes from pass to fail, which is a stronger claim than a higher total.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .answer_key import AnswerKeyItem
from .checks import grade
from .e2e_eval import Run


@dataclass
class Transition:
    id: str
    before: bool
    after: bool
    detail: str = ""


@dataclass
class Comparison:
    before: int
    after: int
    total: int
    fixed: list[Transition] = field(default_factory=list)
    broken: list[Transition] = field(default_factory=list)
    only_in_one: list[str] = field(default_factory=list)

    def render(self, label_before: str = "before", label_after: str = "after") -> str:
        lines = [
            f"{label_before}: {self.before}/{self.total} -> {label_after}: {self.after}/{self.total}",
            f"fail -> pass: {len(self.fixed)}" + (f" ({', '.join(t.id for t in self.fixed)})" if self.fixed else ""),
            f"pass -> fail: {len(self.broken)}" + (f" ({', '.join(t.id for t in self.broken)})" if self.broken else ""),
        ]
        lines += [f"  {t.id}: {t.detail}" for t in self.broken if t.detail]
        if self.only_in_one:
            lines.append(f"not in both runs (ignored): {', '.join(self.only_in_one)}")
        return "\n".join(lines)


def rescore(run: Run, items: Sequence[AnswerKeyItem]) -> Comparison:
    """Stored verdict -> verdict with the current key. Passages are not re-checked: the stored run keeps only ids."""
    by_id = {it.id: it for it in items}
    cmp = Comparison(0, 0, 0)
    for r in run.results:
        item = by_id.get(r.id)
        if item is None:
            cmp.only_in_one.append(r.id)
            continue
        now = grade(item, r.answer)
        cmp.total += 1
        cmp.before += r.passed
        cmp.after += now.passed
        if r.passed != now.passed:
            t = Transition(r.id, r.passed, now.passed, "; ".join(map(str, now.failures)))
            (cmp.fixed if now.passed else cmp.broken).append(t)
    return cmp


def compare(before: Run, after: Run) -> Comparison:
    a = {r.id: r for r in before.results}
    b = {r.id: r for r in after.results}
    cmp = Comparison(0, 0, 0, only_in_one=sorted(set(a) ^ set(b)))
    for qid in [r.id for r in before.results if r.id in b]:
        x, y = a[qid], b[qid]
        cmp.total += 1
        cmp.before += x.passed
        cmp.after += y.passed
        if x.passed != y.passed:
            detail = "; ".join(f"{f['kind']}: {f['detail']}" for f in y.grade["failures"])
            (cmp.fixed if y.passed else cmp.broken).append(Transition(qid, x.passed, y.passed, detail))
    return cmp

"""Consistency checks between the answer key and the corpus, run offline (and in CI).

A key item is only fair if a model that copies the statute would pass it:
- every legal-basis article exists in the corpus;
- for every expected term, at least one alternative appears in the text of the expected articles;
- no forbidden term appears in the text of the expected articles.
"""

from __future__ import annotations

from collections.abc import Sequence

from .answer_key import AnswerKeyItem
from .checks import matching_alternative
from .corpus import Article


def validate(items: Sequence[AnswerKeyItem], articles: Sequence[Article]) -> list[str]:
    by_id = {a.id: a for a in articles}
    problems: list[str] = []
    for it in items:
        missing = [ref for ref in it.legal_basis if ref not in by_id]
        problems += [f"{it.id}: legal basis {ref!r} is not in the corpus" for ref in missing]
        gold = "\n".join(by_id[ref].text for ref in it.legal_basis if ref in by_id).lower()
        if not gold:
            continue
        problems += [
            f"{it.id}: no alternative of expected {spec!r} is in {', '.join(it.legal_basis)}"
            for spec in it.expected
            if matching_alternative(spec, gold) is None
        ]
        problems += [
            f"{it.id}: forbidden {spec!r} appears in {', '.join(it.legal_basis)}"
            for spec in it.forbidden
            if matching_alternative(spec, gold) is not None
        ]
    return problems

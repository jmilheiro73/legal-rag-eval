"""The answer key itself is tested: schema, and consistency with the corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag_eval.answer_key import AnswerKeyItem, ArticleRef, load_answer_key
from legal_rag_eval.corpus import load_corpus
from legal_rag_eval.validate import validate


def test_article_ref_parsing() -> None:
    ref = ArticleRef.parse("CLT art. 476-A")
    assert (ref.law, ref.number, ref.normalized, ref.id) == ("CLT", "476-A", "476a", "CLT art. 476-A")
    assert ArticleRef.parse("CPC art. 1.009").normalized == "1009"
    assert ArticleRef.parse("Lei 12.506 art. 1").statute_number == "12506"
    assert ArticleRef.parse("CDC art. 26").statute_number is None
    with pytest.raises(ValueError):
        ArticleRef.parse("artigo 26 do CDC")


def test_item_validation() -> None:
    with pytest.raises(ValueError, match="starts_with"):
        AnswerKeyItem.from_dict({"question": "P?", "expected": ["x"], "starts_with": "talvez"}, "x-01")
    with pytest.raises(ValueError, match="nothing to check"):
        AnswerKeyItem.from_dict({"question": "P?"}, "x-01")
    with pytest.raises(ValueError, match="unknown fields"):
        AnswerKeyItem.from_dict({"question": "P?", "expected": ["x"], "expectd": []}, "x-01")


def test_private_pilot_format_is_accepted(tmp_path: Path) -> None:
    p = tmp_path / "legacy.json"
    p.write_text(
        json.dumps(
            [
                {
                    "pergunta": "Qual o prazo?",
                    "esperado": ["8|oito"],
                    "inicio": "sim",
                    "nao_pode": ["15 dias"],
                    "fonte": "CLT",
                }
            ]
        ),
        encoding="utf-8",
    )
    [it] = load_answer_key(p)
    assert (it.id, it.question, it.expected, it.starts_with, it.forbidden, it.note) == (
        "legacy-01",
        "Qual o prazo?",
        ["8|oito"],
        "sim",
        ["15 dias"],
        "CLT",
    )


def test_duplicated_ids(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    p.write_text(json.dumps([{"id": "a", "question": "P?", "expected": ["x"]}] * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        load_answer_key(p)


def test_public_key_is_consistent_with_the_corpus(key_path: Path, corpus_dir: Path) -> None:
    assert validate(load_answer_key(key_path), load_corpus(corpus_dir)) == []


def test_validate_reports_unanswerable_items(corpus_dir: Path) -> None:
    items = [
        AnswerKeyItem("x-01", "P?", expected=["noventa"], legal_basis=["CDC art. 999"]),
        AnswerKeyItem("x-02", "P?", expected=["quarenta"], legal_basis=["CDC art. 26"]),
        AnswerKeyItem("x-03", "P?", expected=["noventa"], forbidden=["trinta"], legal_basis=["CDC art. 26"]),
    ]
    problems = validate(items, load_corpus(corpus_dir))
    assert len(problems) == 3
    assert "not in the corpus" in problems[0] and "quarenta" in problems[1] and "forbidden 'trinta'" in problems[2]


def test_public_key_shape(key_path: Path) -> None:
    items = load_answer_key(key_path)
    assert 20 <= len(items) <= 30
    assert all(it.legal_basis for it in items)
    assert sum(1 for it in items if it.starts_with) >= 5
    assert sum(1 for it in items if it.forbidden) >= 5
    assert {it.starts_with for it in items if it.starts_with} == {"sim", "não"}

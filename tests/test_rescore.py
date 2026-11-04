from __future__ import annotations

from conftest import item

from legal_rag_eval.checks import grade
from legal_rag_eval.e2e_eval import QuestionResult, Run
from legal_rag_eval.rescore import compare, rescore


def result(qid: str, answer: str, passed: bool) -> QuestionResult:
    failures = [] if passed else [{"kind": "missing_term", "detail": "x"}]
    return QuestionResult(
        qid,
        "P?",
        "CPC",
        {},
        answer,
        1.0,
        {"passed": passed, "failures": failures, "unsupported_citations": None, "cited": []},
    )


def test_stricter_key_turns_false_passes_into_failures() -> None:
    # Under a loose key (just the term) this wrong answer passed: it repeats the question.
    loose = item(id="cpc-03", expected=["dobro"])
    answer = "Não, a Fazenda não tem prazo em dobro."
    assert grade(loose, answer).passed
    strict = item(id="cpc-03", expected=["dobro"], starts_with="sim", legal_basis=["CPC art. 183"])
    run = Run({}, [result("cpc-03", answer, True), result("gone", "x", True)])
    cmp = rescore(run, [strict])
    assert (cmp.before, cmp.after, cmp.total) == (1, 0, 1)
    assert [t.id for t in cmp.broken] == ["cpc-03"] and "yes_no" in cmp.broken[0].detail
    assert cmp.only_in_one == ["gone"]
    assert "stored: 1/1 -> current key: 0/1" in cmp.render("stored", "current key")


def test_compare_lists_transitions_both_ways() -> None:
    a = Run({}, [result("q1", "", False), result("q2", "", True), result("q3", "", True), result("q4", "", False)])
    b = Run({}, [result("q1", "", True), result("q2", "", True), result("q3", "", False), result("q5", "", True)])
    cmp = compare(a, b)
    assert (cmp.before, cmp.after, cmp.total) == (2, 2, 3)
    assert [t.id for t in cmp.fixed] == ["q1"] and [t.id for t in cmp.broken] == ["q3"]
    assert cmp.only_in_one == ["q4", "q5"]
    text = cmp.render("a", "b")
    assert "a: 2/3 -> b: 2/3" in text and "pass -> fail: 1 (q3)" in text


def test_no_regressions_reads_clearly() -> None:
    a = Run({}, [result("q1", "", False), result("q2", "", True)])
    b = Run({}, [result("q1", "", True), result("q2", "", True)])
    assert "pass -> fail: 0" in compare(a, b).render()

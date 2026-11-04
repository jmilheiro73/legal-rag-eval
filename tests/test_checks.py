from __future__ import annotations

import pytest
from conftest import item, passage

from legal_rag_eval.checks import (
    FORBIDDEN,
    LEGAL_BASIS,
    MISSING_NUMBER,
    MISSING_TERM,
    SEPARATOR,
    YES_NO,
    check_expected,
    check_forbidden,
    check_legal_basis,
    check_yes_no,
    first_words,
    grade,
    model_text,
    term_present,
)


@pytest.mark.parametrize(
    ("term", "text", "found"),
    [
        ("15", "o prazo é de 15 dias", True),
        ("15", "prazo de 15 (quinze) dias", True),
        ("15", "art. 1.015 do cpc", False),
        ("15", "150 dias", False),
        ("15", "1,15 por cento", False),
        ("15", "15.000 reais", False),
        ("15", "dia 15.", True),
        ("5", "art. 7º, xxix: 5 anos", True),
        ("2%", "multa de 12% do faturamento", False),
        ("2%", "até 2% do faturamento", True),
        ("50.000.000", "r$ 50.000.000,00 por infração", True),
        ("15", "r$ 15,00", True),
        ("15", "15,5 dias", False),
        ("1.009", "art. 11.009", False),
        ("dobro", "prazo em dobro", True),
        ("úteis", "somente os dias úteis", True),
        ("prorrog", "o prazo pode ser prorrogado", True),
        ("", "qualquer texto", False),
    ],
)
def test_term_present_matches_whole_numbers_and_substrings(term: str, text: str, found: bool) -> None:
    assert term_present(term, text) is found


def test_expected_terms_with_alternatives_and_failure_kinds() -> None:
    text = "O prazo é de quinze dias, nos termos do art. 335 do CPC."
    assert check_expected(text, ["15|quinze", "335"]) == []
    failures = check_expected(text, ["30|trinta", "contestação"])
    assert [(f.kind, f.detail) for f in failures] == [(MISSING_NUMBER, "30|trinta"), (MISSING_TERM, "contestação")]
    assert check_expected("Qualquer coisa", []) == []


def test_expected_terms_are_case_insensitive() -> None:
    assert check_expected("Cabe ao CONTROLADOR comunicar", ["controlador"]) == []


@pytest.mark.parametrize(
    ("answer", "expectation", "ok"),
    [
        ("Sim. A Fazenda Pública tem prazo em dobro.", "sim", True),
        ("Sim, pode.", "sim", True),
        ("A resposta é sim: prazo em dobro.", "sim", True),  # within the first six words
        ("Não. Os prazos são contados em dias úteis.", "não", True),
        ("Não, salvo previsão diversa.", "sim", False),
        ("Sim, os prazos não contam nos feriados.", "não", False),  # has "não", but starts with "sim"
        ("Não sim, depende.", "sim", False),  # starts with the opposite word
        ("Depende do caso concreto e das provas, mas em regra sim.", "sim", False),  # too late
        ("", "sim", False),
    ],
)
def test_yes_no(answer: str, expectation: str, ok: bool) -> None:
    assert (check_yes_no(answer, expectation) == []) is ok


def test_yes_no_is_skipped_without_expectation() -> None:
    assert check_yes_no("Talvez.", None) == []


def test_first_words_ignore_punctuation_and_keep_accents() -> None:
    assert first_words("Não. O prazo, de 15 dias, é contado...") == ["não", "o", "prazo", "de", "dias", "é"]


def test_forbidden_reports_the_alternative_found() -> None:
    failures = check_forbidden("Prazo de 5 anos, totalizando 7 anos.", ["prorrog", "7 anos|sete anos"])
    assert [(f.kind, f.detail) for f in failures] == [(FORBIDDEN, "7 anos")]
    assert check_forbidden("Prazo em dobro.", ["quádruplo"]) == []


@pytest.mark.parametrize(
    ("answer", "basis", "ok"),
    [
        ("Conforme o art. 335 do CPC.", ["CPC art. 335"], True),
        ("Nos termos do artigo 335.", ["CPC art. 335"], True),
        ("O CPC trata disso na contestação.", ["CPC art. 335"], False),
        ("Fundamento: art. 1009.", ["CPC art. 1.009"], True),  # 1.009 == 1009
        ("art. 476A da CLT", ["CLT art. 476-A"], True),
        ("art. 476 da CLT", ["CLT art. 476-A"], False),
        ("art. 186 do Código Civil", ["CC art. 927", "CC art. 186"], True),  # any of them
        ("Lei 12.506/2011", ["Lei 12.506 art. 1"], True),  # a statute named by number can be cited whole
        ("CDC (Lei 8.078/1990)", ["CDC art. 26"], False),  # a code needs the article
        ("Sem citação.", [], True),
    ],
)
def test_legal_basis(answer: str, basis: list[str], ok: bool) -> None:
    failures = check_legal_basis(answer, basis)
    assert (failures == []) is ok
    if not ok:
        assert failures[0].kind == LEGAL_BASIS


def test_only_text_before_the_separator_is_scored() -> None:
    answer = "O prazo é de 15 dias." + SEPARATOR + "**Fontes:** CPC art. 335"
    assert model_text(answer) == "O prazo é de 15 dias."
    g = grade(item(expected=["15"], legal_basis=["CPC art. 335"]), answer)
    assert not g.passed and [f.kind for f in g.failures] == [LEGAL_BASIS]


def test_grade_combines_all_rules() -> None:
    it = item(
        question="A Fazenda Pública tem prazo diferenciado?",
        starts_with="sim",
        expected=["dobro"],
        forbidden=["quádruplo"],
        legal_basis=["CPC art. 183"],
    )
    assert grade(it, "Sim. A Fazenda Pública tem prazo em dobro (art. 183 do CPC).").passed
    g = grade(it, "Sim. Prazo em quádruplo para contestar e em dobro para recorrer.")
    assert not g.passed and [f.kind for f in g.failures] == [FORBIDDEN, LEGAL_BASIS]
    g = grade(it, "Não. O prazo é comum.")
    assert [f.kind for f in g.failures] == [MISSING_TERM, YES_NO, LEGAL_BASIS]


def test_a_wrong_answer_that_repeats_the_question_does_not_pass() -> None:
    it = item(
        question="O fornecedor de serviços só responde se tiver culpa?",
        starts_with="não",
        expected=["independentemente da existência de culpa|independentemente de culpa"],
        legal_basis=["CDC art. 14"],
    )
    assert not grade(it, "Sim, o fornecedor de serviços só responde se tiver culpa (art. 14 do CDC).").passed
    assert grade(it, "Não. Responde independentemente de culpa (CDC, art. 14).").passed


def test_source_check_is_reported_but_does_not_fail_the_answer() -> None:
    it = item(expected=["15"], legal_basis=["CPC art. 335"])
    passages = [passage("CPC art. 335", "[CPC, Lei 13.105/2015] Art. 335. O réu poderá oferecer contestação ... 15")]
    g = grade(it, "Prazo de 15 dias (art. 335 do CPC; ver também art. 231).", passages)
    assert g.passed and g.unsupported_citations == ["art. 231"] and g.cited == ["art. 335", "art. 231"]
    assert grade(it, "15 dias, art. 335.").unsupported_citations is None  # no passages: not checked

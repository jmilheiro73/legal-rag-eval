from __future__ import annotations

from conftest import passage

from legal_rag_eval.citations import check_sources, extract, normalize_number


def test_extracts_articles_sumulas_and_statutes_with_normalised_numbers() -> None:
    cs = extract(
        "Prazo de 15 dias (art. 1.009 do CPC; artigo 1009 de novo), Súmula nº 37 do STJ, "
        "Lei 8.078/90, art. 476-A da CLT"
    )
    assert [c.key for c in cs] == [("art", "1009"), ("sumula", "37"), ("lei", "8078"), ("art", "476a")]
    assert cs[0].text == "art. 1.009" and cs[3].text == "art. 476-A"


def test_suffix_letter_must_be_uppercase() -> None:
    assert extract("art. 476A")[0].key == ("art", "476a")
    assert [c.text for c in extract("arts. 1.029 e 105")] == ["art. 1.029"]  # "e" is not a suffix


def test_ordinal_and_decreto_lei() -> None:
    assert [c.key for c in extract("art. 7º, XXIX, da CF")] == [("art", "7")]
    assert [c.key for c in extract("Decreto-Lei nº 5.452/1943")] == [("lei", "5452")]


def test_no_citation() -> None:
    assert extract("15 dias úteis, sem citação") == []


def test_normalize_number() -> None:
    assert normalize_number("1.009") == "1009"
    assert normalize_number("476", "A") == "476a"


def test_supported_and_unsupported_citations() -> None:
    passages = [
        passage("CPC art. 1.009", "[CPC, Lei 13.105/2015] Art. 1.009. Da sentença cabe apelação."),
        passage("CPC art. 219", "[CPC, Lei 13.105/2015] Art. 219. ... somente os dias úteis."),
    ]
    check = check_sources(
        "Apelação em 15 dias úteis (art. 1009 e art. 219). Ver também art. 1.029 e Súmula 37.", passages
    )
    assert [c.text for c in check.unsupported] == ["art. 1.029", "Súmula 37"]
    assert check.sources == ["CPC art. 1.009", "CPC art. 219"]
    assert not check.ok
    assert check.as_dict()["unsupported"] == ["art. 1.029", "Súmula 37"]


def test_statute_prefix_in_passage_supports_citing_the_statute() -> None:
    p = passage(
        "CDC art. 49", "[CDC, Lei 8.078/1990] Art. 49. O consumidor pode desistir do contrato, no prazo de 7 dias"
    )
    assert check_sources("7 dias, art. 49 da Lei 8.078/1990.", [p]).ok


def test_answer_without_citations_is_ok() -> None:
    check = check_sources("Depende do contrato.", [passage("CPC art. 335", "Art. 335.")])
    assert check.ok and check.as_dict() == {"cited": [], "unsupported": [], "sources": []}

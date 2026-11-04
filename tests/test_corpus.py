from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag_eval.corpus import chunk_article, chunk_corpus, estimate_tokens, load_corpus, parse_corpus_file

FRONT = """---
code: {code}
title: Teste
act: Lei nº 1, de 1º de janeiro de 2000
citation: {citation}
url: https://www.planalto.gov.br/exemplo.htm
retrieved: 2026-11-04
---

"""


def write(tmp_path: Path, body: str, code: str = "TST", citation: str = "Lei 1/2000", name: str = "t.md") -> Path:
    p = tmp_path / name
    p.write_text(FRONT.format(code=code, citation=citation) + body, encoding="utf-8")
    return p


def test_article_numbers_in_planalto_styles(tmp_path: Path) -> None:
    p = write(
        tmp_path,
        "Art. 7º Caput.\nI - um;\n\nArt. 58 - Caput antigo.\n\nArt. 476-A. Caput.\n§ 1º Parágrafo.\n\n"
        "Art. 1.009. Caput.\nParágrafo único. Texto.\n",
    )
    arts = parse_corpus_file(p)
    assert [a.id for a in arts] == ["TST art. 7", "TST art. 58", "TST art. 476-A", "TST art. 1.009"]
    assert arts[0].lines == ("Art. 7º Caput.", "I - um;")
    assert arts[3].text == "Art. 1.009. Caput.\nParágrafo único. Texto."


def test_front_matter_is_required(tmp_path: Path) -> None:
    p = tmp_path / "bad.md"
    p.write_text("Art. 1º Texto.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        parse_corpus_file(p)


def test_text_before_first_article_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before the first article"):
        parse_corpus_file(write(tmp_path, "Preâmbulo solto.\n\nArt. 1º Texto.\n"))


def test_duplicate_article_ids_across_files(tmp_path: Path) -> None:
    write(tmp_path, "Art. 1º A.\n", name="a.md")
    write(tmp_path, "Art. 1º B.\n", name="b.md")
    with pytest.raises(ValueError, match="duplicated"):
        load_corpus(tmp_path)


def test_short_article_is_one_chunk_with_statute_prefix(tmp_path: Path) -> None:
    [art] = parse_corpus_file(
        write(
            tmp_path,
            "Art. 26. O direito caduca em:\nI - trinta dias;\nII - noventa dias.\n",
            code="CDC",
            citation="Lei 8.078/1990",
        )
    )
    [chunk] = chunk_article(art)
    assert chunk.text.startswith("[CDC, Lei 8.078/1990] Art. 26. O direito caduca em:\nI - trinta dias;")
    assert chunk.article_id == "CDC art. 26" and chunk.chunk_id == "CDC art. 26#0" and chunk.embed_text == chunk.text


def test_statute_named_by_number_is_not_repeated_in_prefix(tmp_path: Path) -> None:
    [art] = parse_corpus_file(write(tmp_path, "Art. 1º Texto.\n", code="Lei 12.506", citation="Lei 12.506/2011"))
    assert chunk_article(art)[0].text == "[Lei 12.506/2011] Art. 1º Texto."


def test_long_article_is_split_by_provision_keeping_caput_and_paragraph(tmp_path: Path) -> None:
    incisos = "\n".join(f"{r} - hipótese número {r} " + "palavra " * 30 + ";" for r in ("I", "II", "III", "IV", "V"))
    body = (
        f"Art. 9º O tratamento somente poderá ser realizado nas seguintes hipóteses:\n{incisos}\n[...]\n"
        "§ 2º As atividades consistem em:\nI - aceitar reclamações;\n"
    )
    [art] = parse_corpus_file(write(tmp_path, body))
    chunks = chunk_article(art, max_tokens=200)
    assert len(chunks) == 7  # five incisos, the paragraph, the inciso under the paragraph; "[...]" is skipped
    assert all(c.article_id == "TST art. 9" for c in chunks)
    assert [c.chunk_id for c in chunks][:2] == ["TST art. 9#0", "TST art. 9#1"]
    assert chunks[0].text.startswith("[TST, Lei 1/2000] Art. 9º O tratamento") and "\nI - hipótese" in chunks[0].text
    assert chunks[0].embed_text.startswith("[TST, Lei 1/2000] Art. 9º, I - hipótese")
    assert chunks[-1].text.endswith("§ 2º As atividades consistem em:\nI - aceitar reclamações;")
    assert all(estimate_tokens(c.text) <= 200 for c in chunks)


def test_real_corpus_loads_and_chunks(corpus_dir: Path) -> None:
    articles = load_corpus(corpus_dir)
    assert len(articles) >= 25
    assert all(a.law.url.startswith("https://www.planalto.gov.br/") for a in articles)
    chunks = chunk_corpus(articles)
    assert {c.article_id for c in chunks} == {a.id for a in articles}
    assert len({c.chunk_id for c in chunks}) == len(chunks)

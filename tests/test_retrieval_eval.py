from __future__ import annotations

from pathlib import Path

from conftest import StaticRetriever, item, mock_http, openai_handler, passage

from legal_rag_eval.answer_key import load_answer_key
from legal_rag_eval.clients import EmbeddingClient
from legal_rag_eval.corpus import chunk_corpus, load_corpus
from legal_rag_eval.rag import DenseRetriever
from legal_rag_eval.retrieval_eval import evaluate_retrieval, render_markdown, score_passages
from legal_rag_eval.store import VectorStore, connect


def test_hit_recall_coverage_and_rank() -> None:
    it = item(expected=["dobro"], legal_basis=["CPC art. 183", "CPC art. 219"])
    passages = [
        passage("CPC art. 335", "contestação em 15 dias"),
        passage("CPC art. 183", "gozarão de prazo em dobro"),
        passage("CPC art. 183", "segundo pedaço do mesmo artigo", n=1),
        passage("CPC art. 219", "dias úteis"),
    ]
    r = score_passages(it, passages, [1, 2, 4])
    assert r.ranks == {"CPC art. 183": 2, "CPC art. 219": 4}
    assert (r.hit(1), r.hit(2), r.hit(4)) == (False, True, True)
    assert (r.recall(1), r.recall(2), r.recall(4)) == (0.0, 0.5, 1.0)
    assert r.missing_terms == {1: ["dobro"], 2: [], 4: []}
    assert r.reciprocal_rank() == 0.5


def test_item_without_expected_terms_uses_hit_for_coverage() -> None:
    it = item(starts_with="sim", legal_basis=["CC art. 927"])
    r = score_passages(it, [passage("CC art. 186", "ato ilícito")], [1])
    assert not r.covered(1, has_terms=False)


def test_report_and_misses() -> None:
    items = [
        item(id="a", question="qa", expected=["dobro"], legal_basis=["CPC art. 183"]),
        item(id="b", question="qb", expected=["úteis"], legal_basis=["CPC art. 219"]),
    ]
    retriever = StaticRetriever(
        {"qa": [passage("CPC art. 183", "prazo em dobro")], "qb": [passage("CPC art. 335", "quinze dias")]}
    )
    report = evaluate_retrieval(items, retriever, ks=[1])
    assert report.counts(1) == {"hit": 1, "coverage": 1}
    assert report.summary()[1]["recall"] == 0.5
    md = render_markdown(report)
    assert "| 1 | 1/2 (50%) | 50% | 1/2 (50%) |" in md
    assert "**b** qb" in md and "expected CPC art. 219 not retrieved" in md and "Retrieved: CPC art. 335" in md
    assert report.as_dict()["results"][0]["ranks"] == {"CPC art. 183": 1}


def test_real_corpus_through_in_memory_qdrant(key_path: Path, corpus_dir: Path) -> None:
    """Ingest the public corpus with fake embeddings into an in-memory Qdrant and run the retrieval eval on it.
    Checks the plumbing (ids, payloads, ranking direction), not retrieval quality."""
    http = mock_http(openai_handler())
    embedder = EmbeddingClient("http://embed.test/v1", "fake", http=http, batch_size=7)
    chunks = chunk_corpus(load_corpus(corpus_dir))
    store = VectorStore(connect(path=":memory:"), "test")
    vectors = embedder.embed([c.embed_text for c in chunks])
    store.recreate(len(vectors[0]))
    store.upsert(chunks, vectors, "fake")
    assert store.count() == len(chunks) and store.embed_model() == "fake"

    retriever = DenseRetriever(embedder, store)
    exact = retriever.retrieve(chunks[0].embed_text, 3)
    assert exact[0].chunk_id == chunks[0].chunk_id  # a chunk's own text ranks itself first
    assert exact[0].score >= exact[1].score >= exact[2].score

    items = load_answer_key(key_path)
    report = evaluate_retrieval(items, retriever, ks=[1, 5])
    assert len(report.results) == len(items)
    assert all(len(r.retrieved) == 5 for r in report.results)
    s = report.summary()
    assert 0.0 <= s[1]["hit"] <= s[5]["hit"] <= 1.0

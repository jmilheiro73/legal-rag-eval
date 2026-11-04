from __future__ import annotations

import json
from pathlib import Path

import httpx
from conftest import StaticRetriever, item, mock_http, openai_handler, passage

from legal_rag_eval.clients import ChatClient
from legal_rag_eval.e2e_eval import GENERATION, RETRIEVAL, Run, attribute, load_run, run_local, save_run
from legal_rag_eval.rag import BlackBoxPipeline, RagPipeline, build_prompt, load_template
from legal_rag_eval.report import render_run

ITEMS = [
    item(id="cpc-01", question="Qual o prazo para contestar?", expected=["15|quinze"], legal_basis=["CPC art. 335"]),
    item(
        id="cpc-02",
        question="Os prazos são contados em dias corridos?",
        starts_with="não",
        expected=["úteis"],
        legal_basis=["CPC art. 219"],
    ),
    item(
        id="cpc-03",
        question="A Fazenda tem prazo diferenciado?",
        starts_with="sim",
        expected=["dobro"],
        forbidden=["quádruplo"],
        legal_basis=["CPC art. 183"],
    ),
]
PASSAGES = {
    ITEMS[0].question: [
        passage(
            "CPC art. 335",
            "[CPC, Lei 13.105/2015] Art. 335. O réu poderá oferecer contestação"
            ", por petição, no prazo de 15 (quinze) dias",
        )
    ],
    ITEMS[1].question: [passage("CPC art. 219", "[CPC, Lei 13.105/2015] Art. 219. ... somente os dias úteis.")],
    ITEMS[2].question: [passage("CPC art. 335", "[CPC, Lei 13.105/2015] Art. 335. contestação em 15 dias")],  # miss
}
ANSWERS = {
    ITEMS[0].question: "O prazo é de 15 (quinze) dias, conforme o art. 335 do CPC.",
    ITEMS[1].question: "Sim, são dias corridos.",  # wrong although the passage was retrieved
    ITEMS[2].question: "Sim. Prazo em quádruplo (art. 188 do CPC).",  # wrong, and the passage was not retrieved
}


def scripted(prompt: str) -> str:
    return next(a for q, a in ANSWERS.items() if prompt.rstrip().endswith(q))


def rag(http: httpx.Client) -> RagPipeline:
    chat = ChatClient("http://llm.test/v1", "test-model", http=http)
    return RagPipeline(StaticRetriever(PASSAGES), chat, load_template("basis"), k=5)


def test_prompt_contains_context_and_question() -> None:
    prompt = build_prompt(load_template("basis"), "Pergunta?", PASSAGES[ITEMS[0].question])
    assert '<document source="CPC art. 335">\n[CPC, Lei 13.105/2015] Art. 335.' in prompt
    assert prompt.rstrip().endswith("Pergunta?")
    assert "name the law article" in prompt and "name the law article" not in load_template("no-basis")


def test_run_grades_and_attributes_misses() -> None:
    seen: list[httpx.Request] = []
    results = run_local(ITEMS, rag(mock_http(openai_handler(scripted, seen))), progress=None)
    assert [r.passed for r in results] == [True, False, False]
    assert [r.cause for r in results] == [None, GENERATION, RETRIEVAL]
    assert results[1].grade["failures"] == [
        {"kind": "missing_term", "detail": "úteis"},
        {"kind": "yes_no", "detail": "expected 'não' at the start"},
        {"kind": "legal_basis", "detail": "CPC art. 219"},
    ]
    assert results[2].grade["unsupported_citations"] == ["art. 188"]
    assert results[0].passages == [{"chunk_id": "CPC art. 335#0", "article_id": "CPC art. 335", "score": 0.5}]
    body = json.loads(seen[0].content)
    assert body["temperature"] == 0.0 and body["model"] == "test-model" and body["stream"] is False


def test_attribute() -> None:
    assert attribute(True, ["A"], ["A"]) == (True, None)
    assert attribute(False, ["A"], ["B", "A"]) == (True, GENERATION)
    assert attribute(False, ["A"], ["B"]) == (False, RETRIEVAL)
    assert attribute(False, ["A"], None) == (None, None)


def test_a_failing_request_is_recorded_and_the_run_goes_on() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt.rstrip().endswith(ITEMS[0].question):
            return httpx.Response(500, json={"error": "boom"})
        return openai_handler(scripted)(request)

    results = run_local(ITEMS, rag(mock_http(handler)), progress=None)
    assert results[0].error and results[0].answer.startswith("ERROR") and not results[0].passed
    assert len(results) == 3 and results[1].error is None


def test_blackbox_sends_the_bare_question_and_forwards_traceparent() -> None:
    seen: list[httpx.Request] = []
    chat = ChatClient(
        "http://rag.test/v1", "my-rag", http=mock_http(openai_handler(lambda p: "15 dias, art. 335.", seen))
    )
    out = BlackBoxPipeline(chat).answer(
        "Qual o prazo para contestar?", traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    )
    assert out.answer == "15 dias, art. 335." and out.passages is None
    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "Qual o prazo para contestar?"}]
    assert seen[0].headers["traceparent"].startswith("00-aaaa")
    [r] = run_local(ITEMS[:1], BlackBoxPipeline(chat), progress=None)
    assert r.passed and r.cause is None and r.grade["unsupported_citations"] is None


def test_api_key_is_sent_as_bearer_token() -> None:
    seen: list[httpx.Request] = []
    chat = ChatClient("http://llm.test/v1", "m", api_key="test-key", http=mock_http(openai_handler(None, seen)))
    chat.complete([{"role": "user", "content": "oi"}])
    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_save_load_roundtrip_and_report(tmp_path: Path) -> None:
    results = run_local(ITEMS, rag(mock_http(openai_handler(scripted))), progress=None)
    run = Run({"model": "test-model", "pipeline": "rag"}, results)
    path = save_run(run, tmp_path / "runs" / "r.jsonl")
    loaded = load_run(path)
    assert loaded.meta == run.meta and loaded.results == run.results and loaded.passed == 1

    md = render_run(loaded)
    assert "**Passed: 1/3 (33%)**" in md
    assert "| Retrieval (no expected article in the context) | 1 |" in md
    assert "| Generation (expected article retrieved, answer wrong) | 1 |" in md
    assert "| Forbidden term | 1 |" in md and "| Wrong yes/no | 1 |" in md
    assert "### cpc-03: A Fazenda tem prazo diferenciado?" in md and "- Unsupported citations: art. 188" in md
    assert "> Sim. Prazo em quádruplo (art. 188 do CPC)." in md

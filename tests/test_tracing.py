"""Langfuse is optional: these tests cover the no-op paths and the adapter logic against a fake SDK."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from conftest import StaticRetriever, item, mock_http, openai_handler, passage

from legal_rag_eval.clients import ChatClient
from legal_rag_eval.rag import RagPipeline, load_template
from legal_rag_eval.tracing import langfuse_settings, run_with_langfuse, traceparent

ENV = {"LANGFUSE_HOST": "http://langfuse.test", "LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}
ITEMS = [
    item(id="cpc-01", question="Qual o prazo para contestar?", expected=["15"], legal_basis=["CPC art. 335"]),
    item(id="cpc-02", question="Dias corridos?", starts_with="não", expected=["úteis"], legal_basis=["CPC art. 219"]),
]


def pipeline() -> RagPipeline:
    answers = {"Qual o prazo para contestar?": "15 dias (art. 335 e art. 231).", "Dias corridos?": "Sim."}
    chat = ChatClient(
        "http://llm.test/v1",
        "m",
        http=mock_http(openai_handler(lambda p: next(a for q, a in answers.items() if p.rstrip().endswith(q)))),
    )
    retriever = StaticRetriever([passage("CPC art. 335", "Art. 335. ... 15 (quinze) dias")])
    return RagPipeline(retriever, chat, load_template(), k=5)


def test_settings_need_host_and_both_keys() -> None:
    assert langfuse_settings({}) is None
    assert langfuse_settings({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}) is None
    assert langfuse_settings({**ENV, "LANGFUSE_SECRET_KEY": ""}) is None
    assert langfuse_settings(ENV) == {
        "base_url": "http://langfuse.test",
        "public_key": "pk-test",
        "secret_key": "sk-test",
    }
    alt = {"LANGFUSE_BASE_URL": "http://other.test", "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    assert langfuse_settings(alt) is not None and langfuse_settings(alt)["base_url"] == "http://other.test"  # type: ignore[index]


def test_traceparent() -> None:
    assert traceparent("a" * 32, "b" * 16) == f"00-{'a' * 32}-{'b' * 16}-01"
    assert traceparent(None, "b" * 16) is None


def test_not_configured_is_a_no_op() -> None:
    assert run_with_langfuse(ITEMS, pipeline(), "key", "run", {}, env={}) is None


def test_configured_without_the_package_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", None)  # import langfuse -> ImportError
    assert run_with_langfuse(ITEMS, pipeline(), "key", "run", {}, env=ENV) is None
    assert "not installed" in capsys.readouterr().err


@dataclass
class FakeEvaluation:
    name: str
    value: float
    comment: str | None = None


@dataclass
class FakeDatasetItem:
    id: str
    input: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class FakeLangfuse:
    kwargs: dict[str, Any]
    items: dict[str, FakeDatasetItem] = field(default_factory=dict)
    datasets: list[str] = field(default_factory=list)
    scores: list[list[FakeEvaluation]] = field(default_factory=list)
    experiment: dict[str, Any] = field(default_factory=dict)
    flushed: bool = False

    def create_dataset(self, *, name: str, description: str) -> None:
        if name in self.datasets:
            raise RuntimeError("exists")
        self.datasets.append(name)

    def create_dataset_item(
        self, *, dataset_name: str, id: str, input: str, expected_output: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        self.items[id] = FakeDatasetItem(id, input, expected_output, metadata)

    def get_dataset(self, name: str) -> Any:
        # Langfuse returns items newest first; the adapter must restore the key's order.
        return types.SimpleNamespace(
            items=list(reversed(self.items.values()))
            + [FakeDatasetItem("old", "removed question", {}, {"id": "removed"})]
        )

    def get_current_trace_id(self) -> str:
        return "c" * 32

    def get_current_observation_id(self) -> str:
        return "d" * 16

    def run_experiment(
        self,
        *,
        name: str,
        run_name: str,
        data: list[FakeDatasetItem],
        task: Callable[..., dict[str, Any]],
        evaluators: list[Callable[..., list[FakeEvaluation]]],
        max_concurrency: int,
        metadata: dict[str, str],
    ) -> None:
        self.experiment = {
            "name": name,
            "run_name": run_name,
            "ids": [d.metadata["id"] for d in data],
            "metadata": metadata,
            "max_concurrency": max_concurrency,
        }
        for d in data:
            output = task(item=d)
            for ev in evaluators:
                self.scores.append(ev(input=d.input, output=output, expected_output=d.expected_output))

    def flush(self) -> None:
        self.flushed = True


def test_experiment_with_a_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeLangfuse] = []

    def factory(**kwargs: Any) -> FakeLangfuse:
        created.append(FakeLangfuse(kwargs))
        return created[-1]

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=factory, Evaluation=FakeEvaluation))
    results = run_with_langfuse(ITEMS, pipeline(), "answer_key", "m rag basis", {"model": "m"}, env=ENV)
    assert results is not None and [r.id for r in results] == ["cpc-01", "cpc-02"]
    [lf] = created
    assert lf.kwargs == {"base_url": "http://langfuse.test", "public_key": "pk-test", "secret_key": "sk-test"}
    assert sorted(lf.items) == ["answer_key:cpc-01", "answer_key:cpc-02"]  # stable ids: re-runs upsert
    assert lf.items["answer_key:cpc-02"].expected_output == {
        "expected": ["úteis"],
        "starts_with": "não",
        "legal_basis": ["CPC art. 219"],
    }
    assert lf.experiment["ids"] == ["cpc-01", "cpc-02"] and lf.experiment["max_concurrency"] == 1
    first, second = lf.scores
    assert [(e.name, e.value) for e in first] == [
        ("passed", 1.0),
        ("seconds", first[1].value),
        ("unsupported_citations", 1.0),
    ]
    assert second[0].value == 0.0 and second[0].comment and "yes_no" in second[0].comment
    assert lf.flushed

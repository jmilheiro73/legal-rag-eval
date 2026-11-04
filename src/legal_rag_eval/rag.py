"""The system under test: retrieve, fill the RAG template, generate.

Two pipelines share one interface, so the evaluation code does not care which one it runs:

- ``RagPipeline``: the reference RAG of this repo (Qdrant search + template + any OpenAI-compatible chat model).
  It returns the passages it used, which enables the source check and the retrieval/generation attribution.
- ``BlackBoxPipeline``: sends the question as-is to an endpoint that does its own retrieval (an existing RAG
  backend exposing /v1/chat/completions). Only the answer is graded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from .clients import ChatClient, EmbeddingClient
from .store import Passage, VectorStore

TEMPLATES = {"basis": "rag_template.txt", "no-basis": "rag_template_no_basis.txt"}


def load_template(name_or_path: str = "basis") -> str:
    if name_or_path in TEMPLATES:
        return (
            resources.files("legal_rag_eval").joinpath("prompts", TEMPLATES[name_or_path]).read_text(encoding="utf-8")
        )
    return Path(name_or_path).read_text(encoding="utf-8")


def build_context(passages: list[Passage]) -> str:
    return "\n".join(f'<document source="{p.article_id}">\n{p.text}\n</document>' for p in passages)


def build_prompt(template: str, question: str, passages: list[Passage]) -> str:
    if "{{CONTEXT}}" not in template or "{{QUERY}}" not in template:
        raise ValueError("the template needs {{CONTEXT}} and {{QUERY}}")
    return template.replace("{{CONTEXT}}", build_context(passages)).replace("{{QUERY}}", question)


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[Passage]: ...


class DenseRetriever:
    def __init__(self, embedder: EmbeddingClient, store: VectorStore):
        self.embedder, self.store = embedder, store

    def retrieve(self, question: str, k: int) -> list[Passage]:
        return self.store.search(self.embedder.embed_one(question), k)


@dataclass
class PipelineAnswer:
    answer: str
    seconds: float
    passages: list[Passage] | None = None  # None: the pipeline does not expose what it retrieved
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Pipeline(Protocol):
    name: str

    def answer(self, question: str, traceparent: str | None = None) -> PipelineAnswer: ...


def _trace_headers(traceparent: str | None) -> dict[str, str]:
    return {"traceparent": traceparent} if traceparent else {}


class RagPipeline:
    name = "rag"

    def __init__(self, retriever: Retriever, chat: ChatClient, template: str, k: int = 5):
        self.retriever, self.chat, self.template, self.k = retriever, chat, template, k

    def answer(self, question: str, traceparent: str | None = None) -> PipelineAnswer:
        t0 = time.perf_counter()
        passages: list[Passage] = []
        try:
            passages = self.retriever.retrieve(question, self.k)
            prompt = build_prompt(self.template, question, passages)
            res = self.chat.complete([{"role": "user", "content": prompt}], headers=_trace_headers(traceparent))
            return PipelineAnswer(res.text, time.perf_counter() - t0, passages, res.usage)
        except Exception as e:  # one failing question must not stop a 30-minute run
            return PipelineAnswer(f"ERROR: {e!r}", time.perf_counter() - t0, passages, error=repr(e))


class BlackBoxPipeline:
    name = "blackbox"

    def __init__(self, chat: ChatClient):
        self.chat = chat

    def answer(self, question: str, traceparent: str | None = None) -> PipelineAnswer:
        t0 = time.perf_counter()
        try:
            res = self.chat.complete([{"role": "user", "content": question}], headers=_trace_headers(traceparent))
            return PipelineAnswer(res.text, time.perf_counter() - t0, None, res.usage)
        except Exception as e:
            return PipelineAnswer(f"ERROR: {e!r}", time.perf_counter() - t0, None, error=repr(e))

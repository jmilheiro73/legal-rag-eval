"""Shared test helpers. Nothing here needs a model, a GPU, a network or a Qdrant server: embeddings are a
deterministic bag-of-words hash, chat answers are scripted, and Qdrant runs in memory."""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from legal_rag_eval.answer_key import AnswerKeyItem
from legal_rag_eval.store import Passage

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "data" / "answer_key.json"
CORPUS = ROOT / "data" / "corpus"
DIM = 64


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        v[zlib.crc32(word.encode()) % dim] += 1.0
    if not any(v):
        v[0] = 1.0
    return v


def openai_handler(
    answer: Callable[[str], str] | None = None, seen: list[httpx.Request] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """An OpenAI-compatible server: /embeddings with fake_vector, /chat/completions with a scripted answer."""

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": fake_vector(t)} for i, t in enumerate(body["input"])]
            return httpx.Response(200, json={"data": data})
        if request.url.path.endswith("/chat/completions"):
            prompt = body["messages"][-1]["content"]
            text = answer(prompt) if answer else "Não encontrei essa informação nos documentos."
            return httpx.Response(
                200, json={"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": len(prompt) // 4}}
            )
        return httpx.Response(404)

    return handle


def mock_http(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def passage(article_id: str, text: str, score: float = 0.5, n: int = 0) -> Passage:
    return Passage(chunk_id=f"{article_id}#{n}", article_id=article_id, text=text, score=score)


def item(**kwargs: object) -> AnswerKeyItem:
    data: dict[str, object] = {"id": "t-01", "question": "Pergunta?"}
    data.update(kwargs)
    return AnswerKeyItem.from_dict(data, "t-01")


class StaticRetriever:
    def __init__(self, passages: dict[str, list[Passage]] | list[Passage]):
        self.passages = passages

    def retrieve(self, question: str, k: int) -> list[Passage]:
        found = self.passages if isinstance(self.passages, list) else self.passages.get(question, [])
        return found[:k]


@pytest.fixture
def key_path() -> Path:
    return KEY


@pytest.fixture
def corpus_dir() -> Path:
    return CORPUS

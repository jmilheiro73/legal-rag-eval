"""Clients for OpenAI-compatible HTTP endpoints (/v1/embeddings and /v1/chat/completions).

That one format covers llama.cpp's llama-server, Ollama, vLLM, LM Studio and most hosted APIs, so the harness has
no provider SDK. Pass an ``httpx.Client`` to reuse connections or, in tests, a mock transport.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


def _headers(api_key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra or {})
    return headers


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        batch_size: int = 16,
        timeout: float = 300.0,
    ):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.http = http or httpx.Client(timeout=timeout)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = self.http.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                headers=_headers(self.api_key),
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            if len(data) != len(batch):
                raise RuntimeError(f"embeddings endpoint returned {len(data)} vectors for {len(batch)} inputs")
            vectors.extend([list(map(float, d["embedding"])) for d in data])
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@dataclass
class ChatResult:
    text: str
    seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        timeout: float = 600.0,
    ):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.http = http or httpx.Client(timeout=timeout)
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, messages: list[dict[str, str]], headers: dict[str, str] | None = None) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if self.temperature is not None:
            body["temperature"] = self.temperature  # 0 by default: the same prompt gives the same answer
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=_headers(self.api_key, headers))
        r.raise_for_status()
        data = r.json()
        return ChatResult(
            data["choices"][0]["message"]["content"] or "", time.perf_counter() - t0, data.get("usage") or {}
        )

"""Settings from environment variables (see .env.example); command-line options override them."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class Settings:
    llm_base_url: str = "http://localhost:8080/v1"  # llama.cpp llama-server default
    llm_model: str = ""
    llm_api_key: str | None = None
    embed_base_url: str = "http://localhost:11434/v1"  # Ollama default
    embed_model: str = "bge-m3"
    embed_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str | None = None  # embedded local mode instead of a server
    collection: str = "legal_rag_eval"
    top_k: int = 5
    temperature: float | None = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        d = cls()
        temperature = env.get("TEMPERATURE", "0")
        return cls(
            llm_base_url=env.get("LLM_BASE_URL") or d.llm_base_url,
            llm_model=env.get("LLM_MODEL") or d.llm_model,
            llm_api_key=env.get("LLM_API_KEY") or None,
            embed_base_url=env.get("EMBED_BASE_URL") or d.embed_base_url,
            embed_model=env.get("EMBED_MODEL") or d.embed_model,
            embed_api_key=env.get("EMBED_API_KEY") or None,
            qdrant_url=env.get("QDRANT_URL") or d.qdrant_url,
            qdrant_path=env.get("QDRANT_PATH") or None,
            collection=env.get("COLLECTION") or d.collection,
            top_k=int(env.get("TOP_K") or d.top_k),
            temperature=None if temperature.lower() in ("", "none") else float(temperature),
        )

"""Qdrant collection for the corpus: create, upsert, search. Dense vectors only, cosine distance.

``QDRANT_URL`` points at a server (see docker-compose.yml); ``QDRANT_PATH`` uses qdrant-client's embedded local mode
instead, which needs no server; ``":memory:"`` is what the tests use.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .corpus import CHUNKER_VERSION, Chunk


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    article_id: str
    text: str
    score: float

    @property
    def source(self) -> str:
        return self.article_id


def connect(url: str | None = None, path: str | None = None) -> QdrantClient:
    if path:
        return QdrantClient(location=":memory:") if path == ":memory:" else QdrantClient(path=path)
    return QdrantClient(url=url or "http://localhost:6333")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"legal-rag-eval:{chunk_id}"))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client, self.collection = client, collection

    def close(self) -> None:
        self.client.close()

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def recreate(self, dim: int) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], embed_model: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=list(v),
                payload={
                    "chunk_id": c.chunk_id,
                    "article_id": c.article_id,
                    "law": c.law,
                    "text": c.text,
                    "embed_model": embed_model,
                    "chunker_version": CHUNKER_VERSION,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def count(self) -> int:
        return int(self.client.count(self.collection).count)

    def embed_model(self) -> str | None:
        """The embedding model the collection was built with (from any point), to catch a mismatch at query time."""
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=["embed_model"], with_vectors=False)
        return str(points[0].payload["embed_model"]) if points and points[0].payload else None

    def search(self, vector: Sequence[float], k: int) -> list[Passage]:
        res = self.client.query_points(self.collection, query=list(vector), limit=k, with_payload=True)
        return [
            Passage(
                chunk_id=str(p.payload["chunk_id"]),
                article_id=str(p.payload["article_id"]),
                text=str(p.payload["text"]),
                score=float(p.score),
            )
            for p in res.points
            if p.payload
        ]

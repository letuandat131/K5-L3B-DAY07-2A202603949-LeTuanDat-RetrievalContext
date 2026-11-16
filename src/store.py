"""Persistent Qdrant vector store with safe metadata filtering."""
import math
from typing import Any
import uuid

from qdrant_client import QdrantClient, models

from .models import ChunkRecord


class QdrantChunkStore:
    def __init__(self, storage_path: str, collection: str, dimensions: int = 384) -> None:
        self.collection = collection
        self.dimensions = dimensions
        self.client = QdrantClient(path=storage_path)

    def _validate_vector(self, vec: list[float]) -> list[float]:
        if len(vec) != self.dimensions or not all(math.isfinite(float(v)) for v in vec) or not any(vec):
            raise ValueError("Invalid embedding vector")
        return [float(v) for v in vec]

    def upsert(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors length mismatch")
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )
        old: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=False, with_vectors=False
            )
            old.update(str(r.id) for r in records)
            if offset is None:
                break
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector=self._validate_vector(v),
                payload={
                    **c.metadata,
                    "id": c.id,
                    "source_id": c.source_id,
                    "source": c.source,
                    "page": c.page,
                    "text": c.text,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start : start + 128], wait=True)
        obsolete = old - {str(p.id) for p in points}
        if obsolete:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=list(obsolete)),
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_keys = {"chunk_type", "chapter", "page", "visual_type"}
        conditions = []
        for k, v in (filters or {}).items():
            if k not in allowed_keys or type(v) not in (str, int):
                raise ValueError(f"Unsupported filter key/value: {k}")
            conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        hits = self.client.query_points(
            self.collection,
            query=self._validate_vector(query_vector),
            limit=top_k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        ).points
        return [
            {
                "id": h.payload["id"],
                "source_id": h.payload["source_id"],
                "source": h.payload["source"],
                "page": h.payload["page"],
                "text": h.payload["text"],
                "score": float(h.score),
                "strategy": "dense",
                "metadata": {
                    k: val
                    for k, val in h.payload.items()
                    if k not in ("id", "source_id", "source", "page", "text")
                },
            }
            for h in hits
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()

"""Multiple retrieval methods (BM25, Dense, Hybrid RRF, Rerank) and context packing."""
import re
from typing import Any
import unicodedata

from rank_bm25 import BM25Okapi

from .models import ChunkRecord


def tokenize(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", norm)


def _match_filters(chunk: ChunkRecord, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if k == "page":
            if chunk.page != v:
                return False
        elif chunk.metadata.get(k) != v:
            return False
    return True


class BM25Retriever:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize(query))
        scored = []
        for chunk, score in zip(self.chunks, scores):
            if score > 0 and _match_filters(chunk, filters):
                scored.append({**chunk.to_day7_dict(), "score": float(score), "strategy": "bm25"})
        scored.sort(key=lambda item: (-item["score"], item["id"]))
        return scored[:top_k]


class DenseRetriever:
    def __init__(self, embedder: Any, store: Any) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        qvec = self.embedder.embed_query(query)
        return self.store.search(qvec, top_k=top_k, filters=filters)


class HybridRRFRetriever:
    """Reciprocal Rank Fusion combining dense semantic retrieval and BM25 lexical retrieval."""

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: BM25Retriever,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        pool_k = max(top_k * 3, 15)
        dense_hits = self.dense.search(query, filters=filters, top_k=pool_k)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=pool_k)

        by_id: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}
        for rank, item in enumerate(dense_hits, 1):
            by_id[item["id"]] = item
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + self.dense_weight / (self.rrf_k + rank)
        for rank, item in enumerate(bm25_hits, 1):
            by_id.setdefault(item["id"], item)
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + self.bm25_weight / (self.rrf_k + rank)

        fused = []
        for cid, score in rrf_scores.items():
            doc = dict(by_id[cid])
            doc["score"] = float(score)
            doc["strategy"] = "hybrid_rrf"
            fused.append(doc)
        fused.sort(key=lambda item: (-item["score"], item["id"]))
        return fused[:top_k]


class HybridRerankRetriever:
    """Multi-stage retriever: wide Hybrid RRF pool + exact ref/caption matching & rerank."""

    def __init__(self, hybrid_retriever: HybridRRFRetriever, candidate_k: int = 35) -> None:
        self.hybrid = hybrid_retriever
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        candidates = self.hybrid.search(query, filters=filters, top_k=self.candidate_k)
        q_tokens = set(tokenize(query))
        refs = re.findall(r"\b(?:Table|Figure|Map|Box)\s+[A-Z0-9]+(?:\.[A-Z0-9]+)*\b", query, re.IGNORECASE)
        if refs:
            seen = {c["id"] for c in candidates}
            for chunk in self.hybrid.bm25.chunks:
                if chunk.id in seen or not _match_filters(chunk, filters):
                    continue
                if any(ref.lower() in chunk.text.lower() for ref in refs):
                    candidates.append({**chunk.to_day7_dict(), "score": 0.01, "strategy": "hybrid_rerank"})
                    seen.add(chunk.id)
        reranked = []
        for item in candidates:
            text_lower = item["text"].lower()
            c_tokens = set(tokenize(item["text"]))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            ref_bonus = 0.25 if any(ref.lower() in text_lower for ref in refs) else 0.0
            type_bonus = 0.06 if refs and item.get("metadata", {}).get("chunk_type") in ("table", "visual") else 0.0
            score = float(item["score"]) + 0.08 * overlap + ref_bonus + type_bonus
            updated = dict(item)
            updated["score"] = score
            updated["strategy"] = "hybrid_rerank"
            reranked.append(updated)
        reranked.sort(key=lambda x: (-x["score"], x["id"]))
        return reranked[:top_k]


def expand_with_neighbors(
    hits: list[dict[str, Any]],
    all_chunks: list[ChunkRecord],
    *,
    max_total: int = 10,
) -> list[dict[str, Any]]:
    """Context engineering: expand adjacent paragraphs and same-page prose for top hits."""
    by_page: dict[int, list[ChunkRecord]] = {}
    for c in all_chunks:
        if c.metadata.get("chunk_type") == "text":
            by_page.setdefault(c.page, []).append(c)
    expanded = list(hits)
    seen = {h["id"] for h in expanded}
    for top in hits[:3]:
        siblings = by_page.get(top["page"], [])
        if not siblings:
            continue
        if top.get("metadata", {}).get("chunk_type") == "text":
            pos = next((i for i, c in enumerate(siblings) if c.id == top["id"]), -1)
            candidates = [siblings[pos + off] for off in (-1, 1, 2) if pos >= 0 and 0 <= pos + off < len(siblings)]
        else:
            # Visual/table hits often rely on surrounding prose on the same page (first & trailing paragraphs).
            candidates = siblings[:1] + (siblings[-2:] if len(siblings) > 1 else [])
        for sib in candidates:
            if sib.id not in seen and len(expanded) < max_total:
                expanded.append({**sib.to_day7_dict(), "score": top["score"] * 0.95, "strategy": "neighbor_expand"})
                seen.add(sib.id)
    expanded.sort(key=lambda x: (-x["score"], x["id"]))
    return expanded[:max_total]


def pack_context(results: list[dict[str, Any]], budget: int = 2400) -> tuple[str, list[str], list[str]]:
    """Deduplicate retrieved chunks and pack up to budget characters, tracking visual PNG paths."""
    blocks: list[str] = []
    ids: list[str] = []
    images: list[str] = []
    seen_ids: set[str] = set()

    for c in results:
        cid = c["id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        block = f"[{cid} | page {c['page']} | {c.get('metadata', {}).get('chunk_type', 'text')}]\n{c['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget or not blocks:
            # Always keep at most budget if a single huge table slightly exceeds small test budget unless empty
            if len(candidate_text) <= budget:
                blocks.append(block)
                ids.append(cid)
                img = c.get("metadata", {}).get("image_path")
                if img and img not in images:
                    images.append(img)
    return "\n\n".join(blocks), ids, images

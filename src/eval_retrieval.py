"""Freeze benchmark and evaluate retrieval methods + context engineering on Dev split."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from .embeddings import LocalMiniLMEmbedder
from .models import ChunkRecord
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    HybridRerankRetriever,
    expand_with_neighbors,
    pack_context,
)
from .store import QdrantChunkStore

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_advanced_benchmark(questions: list[dict], pages_count: int) -> None:
    if pages_count < 100:
        raise ValueError(f"Need >= 100 pages for advanced benchmark, got {pages_count}")
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    if len(dev) < 18 or sum(q["answerable"] for q in dev) < 15 or sum(not q["answerable"] for q in dev) < 3:
        raise ValueError("Dev must contain >= 15 answerable and >= 3 unanswerable questions")
    reqs = {"table": 6, "diagram_image": 5, "chart": 5, "multi_hop": 4, "text": 4, "unanswerable": 6}
    for cat, minimum in reqs.items():
        got = sum(q["category"] == cat for q in test)
        if got < minimum:
            raise ValueError(f"Test category {cat} requires >= {minimum}, got {got}")


def infer_query_filter(question_text: str) -> dict | None:
    """Safely infer coarse chunk_type filter purely from user question wording (never from gold)."""
    low = question_text.lower()
    if re.search(r"\btable\s+[a-z0-9.]+", low):
        return {"chunk_type": "table"}
    if re.search(r"\b(?:figure|map)\s+[a-z0-9.]+", low):
        return {"chunk_type": "visual"}
    return None


def chunk_matches_gold_item(chunk: dict, g: dict) -> bool:
    if chunk["source_id"] != g["source_id"] or chunk["page"] != g["page"]:
        return False
    ev = " ".join(g["evidence"].lower().split())
    txt = " ".join(chunk["text"].lower().split())
    return ev in txt or bool(re.match(r"^(table|figure|map)\s+[a-z0-9.]+", ev) and ev in txt)


def evaluate_hit_and_context(question: dict, hits: list[dict], context_ids: list[str]) -> tuple[bool, bool, bool]:
    """Return (hit1, hit3, context_hit) supporting single-page and multi-hop all-page coverage."""
    golds = question["gold"]
    if not golds:
        return False, False, False
    hit1 = any(chunk_matches_gold_item(hits[0], g) for g in golds) if hits else False
    # For multi_hop, Hit@3 requires at least one gold match in top-3, while Context Hit requires ALL gold pages covered!
    hit3 = any(chunk_matches_gold_item(h, g) for h in hits[:3] for g in golds)
    packed = [h for h in hits if h["id"] in set(context_ids)]
    if question["category"] == "multi_hop":
        context_hit = all(any(chunk_matches_gold_item(h, g) for h in packed) for g in golds)
    else:
        context_hit = any(chunk_matches_gold_item(h, g) for h in packed for g in golds)
    return hit1, hit3, context_hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    validate_advanced_benchmark(questions, manifest["indexed_content_pages"])

    freeze_path = ROOT / "benchmark/freeze.json"
    hashes = {
        "questions": canonical_hash(ROOT / "benchmark/questions.json"),
        "manifest": canonical_hash(ROOT / "data/manifest.json"),
        "config": canonical_hash(ROOT / "configs/experiment.json"),
    }
    if not freeze_path.exists():
        freeze_path.write_text(
            json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        saved = json.loads(freeze_path.read_text(encoding="utf-8"))
        if saved["hashes"] != hashes:
            raise ValueError("Frozen benchmark/manifest/config changed after freeze")

    if args.freeze_only:
        print("Benchmark frozen cleanly:", hashes)
        return

    raw_chunks = json.loads((ROOT / "data/processed/chunks.json").read_text(encoding="utf-8"))
    chunks = [ChunkRecord(**c) for c in raw_chunks]

    bm25 = BM25Retriever(chunks)
    embedder = LocalMiniLMEmbedder(ROOT)
    store = QdrantChunkStore(str(ROOT / config["store"]["path"]), config["store"]["collection"], embedder.dimensions)
    dense = DenseRetriever(embedder, store)
    hybrid_rrf = HybridRRFRetriever(dense, bm25)
    hybrid_rerank = HybridRerankRetriever(hybrid_rrf, candidate_k=35)

    configs = [
        {"name": "1_bm25_b800", "retriever": bm25, "top_k": 3, "budget": 800, "use_filter": False, "expand": False},
        {"name": "2_dense_b800", "retriever": dense, "top_k": 3, "budget": 800, "use_filter": False, "expand": False},
        {"name": "3_hybrid_rrf_b800", "retriever": hybrid_rrf, "top_k": 3, "budget": 800, "use_filter": False, "expand": False},
        {"name": "4_hybrid_rerank_b2400", "retriever": hybrid_rerank, "top_k": 5, "budget": 2400, "use_filter": False, "expand": False},
        {"name": "5_hybrid_rerank_neighbor_b3600", "retriever": hybrid_rerank, "top_k": 6, "budget": 3600, "use_filter": True, "expand": True},
    ]

    split_questions = [q for q in questions if q["split"] == args.split and q["answerable"]]
    summaries = []
    out_dir = ROOT / "results" / f"retrieval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for cfg in configs:
            traces = []
            for q in split_questions:
                flt = infer_query_filter(q["question"]) if cfg["use_filter"] else None
                t0 = time.perf_counter()
                if flt:
                    # Two-lane retrieval: filtered visual/table lane + unfiltered general lane so multi-hop prose isn't dropped.
                    filtered_hits = cfg["retriever"].search(q["question"], filters=flt, top_k=3)
                    general_hits = cfg["retriever"].search(q["question"], filters=None, top_k=cfg["top_k"])
                    merged: dict[str, dict] = {}
                    for item in filtered_hits + general_hits:
                        if item["id"] not in merged or item["score"] > merged[item["id"]]["score"]:
                            merged[item["id"]] = item
                    hits = sorted(merged.values(), key=lambda x: (-x["score"], x["id"]))[: cfg["top_k"]]
                else:
                    hits = cfg["retriever"].search(q["question"], filters=None, top_k=cfg["top_k"])
                if cfg["expand"]:
                    hits = expand_with_neighbors(hits, chunks, max_total=cfg["top_k"] + 3)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                ctx, ids, imgs = pack_context(hits, budget=cfg["budget"])
                h1, h3, ctx_hit = evaluate_hit_and_context(q, hits, ids)
                traces.append(
                    {
                        "query_id": q["id"],
                        "category": q["category"],
                        "filter_used": flt,
                        "hit1": h1,
                        "hit3": h3,
                        "context_hit": ctx_hit,
                        "context_chars": len(ctx),
                        "context_ids": ids,
                        "visual_images": imgs,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "top_hits": [{"id": h["id"], "page": h["page"], "type": h["metadata"]["chunk_type"], "score": round(h["score"], 4)} for h in hits],
                    }
                )
            n = len(traces)
            s = {
                "config": cfg["name"],
                "split": args.split,
                "n": n,
                "hit1": f"{sum(t['hit1'] for t in traces)}/{n}",
                "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
                "hit3": f"{sum(t['hit3'] for t in traces)}/{n}",
                "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
                "context_evidence": f"{sum(t['context_hit'] for t in traces)}/{n}",
                "context_evidence_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
                "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
                "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
            }
            summaries.append(s)
            (out_dir / f"{cfg['name']}_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        store.close()

    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run the real Multimodal Tool-Calling Agent on dev or held-out test split."""
import argparse
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from .agent import OpenAIVisionController, VertexGeminiVisionController, load_page_image_b64, run_multimodal_agent
from .embeddings import LocalMiniLMEmbedder
from .eval_retrieval import infer_query_filter
from .models import ChunkRecord
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    HybridRerankRetriever,
    expand_with_neighbors,
)
from .store import QdrantChunkStore

ROOT = Path(__file__).resolve().parents[1]


class ConfiguredRetriever:
    """Adapter exposing a unified search interface for baseline or improved retrieval."""

    def __init__(self, mode: str, chunks: list[ChunkRecord], dense: DenseRetriever, bm25: BM25Retriever) -> None:
        self.mode = mode
        self.chunks = chunks
        self.dense = dense
        self.hybrid = HybridRRFRetriever(dense, bm25)
        self.rerank = HybridRerankRetriever(self.hybrid, candidate_k=35)

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 5) -> list[dict]:
        if self.mode == "baseline_dense":
            return self.dense.search(query, filters=filters, top_k=top_k)
        inferred = filters or infer_query_filter(query)
        if inferred:
            lane_a = self.rerank.search(query, filters=inferred, top_k=3)
            lane_b = self.rerank.search(query, filters=None, top_k=top_k)
            merged: dict[str, dict] = {}
            for item in lane_a + lane_b:
                if item["id"] not in merged or item["score"] > merged[item["id"]]["score"]:
                    merged[item["id"]] = item
            hits = sorted(merged.values(), key=lambda x: (-x["score"], x["id"]))[:top_k]
        else:
            hits = self.rerank.search(query, filters=None, top_k=top_k)
        return expand_with_neighbors(hits, self.chunks, max_total=top_k + 3)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--mode", default="improved", choices=("baseline_dense", "improved"))
    parser.add_argument("--provider", default="auto", choices=("auto", "vertex_gemini", "openai"))
    parser.add_argument("--model", default="")
    parser.add_argument("--max-searches", type=int, default=5)
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    questions = [q for q in json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8")) if q["split"] == args.split]
    chunks = [ChunkRecord(**c) for c in json.loads((ROOT / "data/processed/chunks.json").read_text(encoding="utf-8"))]

    provider = args.provider
    if provider == "auto":
        provider = "vertex_gemini" if os.getenv("VERTEX_API_KEY", "").strip() else "openai"
    if provider == "vertex_gemini":
        model_name = args.model or os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
        controller = VertexGeminiVisionController(model=model_name)
    else:
        model_name = args.model or "gpt-4o-mini"
        controller = OpenAIVisionController(model=model_name)
    bm25 = BM25Retriever(chunks)
    embedder = LocalMiniLMEmbedder(ROOT)
    store = QdrantChunkStore(str(ROOT / config["store"]["path"]), config["store"]["collection"], embedder.dimensions)
    dense = DenseRetriever(embedder, store)
    retriever = ConfiguredRetriever(args.mode, chunks, dense, bm25)

    budget = 800 if args.mode == "baseline_dense" else 6400
    top_k = 3 if args.mode == "baseline_dense" else 8

    traces = []
    started = time.perf_counter()
    try:
        for q in questions:
            t0 = time.perf_counter()
            try:
                res = run_multimodal_agent(
                    q["question"],
                    controller,
                    retriever,
                    image_loader=lambda rel, region="full": load_page_image_b64(ROOT / rel, region=region),
                    max_searches=args.max_searches,
                    top_k=top_k,
                    budget=budget,
                )
            except Exception as exc:
                res = {"status": "error", "answer": "", "citations": [], "events": [{"event": "error", "detail": f"{type(exc).__name__}: {exc}"}]}
            traces.append(
                {
                    "query_id": q["id"],
                    "split": q["split"],
                    "category": q["category"],
                    "answerable": q["answerable"],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                    **res,
                }
            )
    finally:
        store.close()

    out_dir = ROOT / "results" / f"agent_{args.split}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "mode": args.mode,
        "provider": provider,
        "split": args.split,
        "model": model_name,
        "n": len(traces),
        "answered": sum(t["status"] == "answered" for t in traces),
        "abstained": sum(t["status"] == "abstain" for t in traces),
        "errors": sum(t["status"] == "error" for t in traces),
        "llm_usage": controller.usage,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

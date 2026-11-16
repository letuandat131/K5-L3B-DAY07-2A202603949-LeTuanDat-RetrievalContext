"""Build the complete WDR 2024 text, table, visual, and Qdrant dense index."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .chunking import chunk_pages
from .embeddings import LocalMiniLMEmbedder
from .ingestion import build_corpus
from .store import QdrantChunkStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-render", action="store_true", help="Re-render page PNG crops cleanly at current render scale.")
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))

    pages, visuals, manifest = build_corpus(ROOT, render_images=True, force_render=args.force_render)
    chunks = chunk_pages(pages, visuals, max_chars=int(config["chunking"]["max_chars"]))

    chunks_path = ROOT / "data/processed/chunks.json"
    chunks_path.write_text(
        json.dumps([c.to_day7_dict() for c in chunks], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    embedder = LocalMiniLMEmbedder(ROOT)
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = QdrantChunkStore(
        str(ROOT / config["store"]["path"]),
        config["store"]["collection"],
        dimensions=embedder.dimensions,
    )
    try:
        store.upsert(chunks, vectors)
        indexed_count = store.count()
    finally:
        store.close()

    by_type = {}
    for c in chunks:
        t = c.metadata["chunk_type"]
        by_type[t] = by_type.get(t, 0) + 1

    summary = {
        "indexed_content_pages": manifest["indexed_content_pages"],
        "physical_pages": manifest["physical_pages"],
        "category_counts": manifest["category_counts"],
        "chunk_count": len(chunks),
        "chunks_by_type": by_type,
        "qdrant_points": indexed_count,
        "embedding_usage": embedder.usage,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    (ROOT / "results/index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

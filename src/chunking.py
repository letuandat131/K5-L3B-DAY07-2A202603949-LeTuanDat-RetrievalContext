"""Multi-granularity chunking for text paragraphs, structured tables, and visual figures/maps."""
import hashlib
import re

from .models import ChunkRecord, ExtractedVisual, PageRecord


def split_by_paragraph(text: str, max_chars: int = 850) -> list[tuple[int, int, str]]:
    """Split normalized page text along paragraph or sentence boundaries up to max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    ends = [m.end() for m in re.finditer(r"\n\n", text)] + [len(text)]
    spans: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        candidates = [e for e in ends if start < e <= limit]
        end = max(candidates) if candidates else limit
        if not candidates and limit < len(text):
            sent = list(re.finditer(r"(?<=[.!?])\s+", text[start:limit]))
            if sent:
                end = start + sent[-1].end()
            else:
                ws = text.rfind(" ", start, limit)
                if ws > start:
                    end = ws + 1
        raw = text[start:end]
        s = start + len(raw) - len(raw.lstrip())
        e = end - (len(raw) - len(raw.rstrip()))
        if s < e and text[s:e].strip():
            spans.append((s, e, text[s:e]))
        start = end
    return spans


def _chunk_id(prefix: str, source_id: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{source_id}:{page}:{idx}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-p{page:03d}-{idx:02d}-{digest[:8]}"


def chunk_pages(
    pages: list[PageRecord],
    visuals: list[ExtractedVisual],
    max_chars: int = 850,
) -> list[ChunkRecord]:
    """Build text, structured table, and visual asset chunks preserving page and image provenance."""
    chunks: list[ChunkRecord] = []
    for page in pages:
        for idx, (s, e, body) in enumerate(split_by_paragraph(page.text, max_chars=max_chars), 1):
            chunks.append(
                ChunkRecord(
                    id=_chunk_id("txt", page.source_id, page.page, idx, body),
                    source_id=page.source_id,
                    source=page.source,
                    page=page.page,
                    text=body,
                    metadata={
                        "chunk_type": "text",
                        "chapter": page.chapter,
                        "printed_page": page.printed_page,
                        "image_path": page.image_path,
                        "start_char": s,
                        "end_char": e,
                    },
                )
            )
        table_caps = [cap for cap in page.visual_captions if cap.lower().startswith("table")]
        for idx, md in enumerate(page.tables_markdown, 1):
            cap = table_caps[idx - 1] if idx - 1 < len(table_caps) else f"Table on PDF page {page.page}"
            full_text = f"{cap}\n{md}"
            chunks.append(
                ChunkRecord(
                    id=_chunk_id("tbl", page.source_id, page.page, idx, full_text),
                    source_id=page.source_id,
                    source=page.source,
                    page=page.page,
                    text=full_text,
                    metadata={
                        "chunk_type": "table",
                        "chapter": page.chapter,
                        "printed_page": page.printed_page,
                        "image_path": page.image_path,
                        "caption": cap,
                    },
                )
            )

    for idx, vis in enumerate(visuals, 1):
        if vis.visual_type == "table":
            continue
        vis_text = f"[{vis.visual_type.upper()}] {vis.caption}\nPage context: {vis.text_summary}"
        chunks.append(
            ChunkRecord(
                id=_chunk_id("vis", vis.source_id, vis.page, idx, vis_text),
                source_id=vis.source_id,
                source=f"https://openknowledge.worldbank.org/bitstreams/862729db-69a7-48a8-ba91-df8fe8d8745b/download",
                page=vis.page,
                text=vis_text,
                metadata={
                    "chunk_type": "visual",
                    "visual_type": vis.visual_type,
                    "visual_id": vis.visual_id,
                    "chapter": vis.chapter,
                    "printed_page": vis.printed_page,
                    "image_path": vis.image_path,
                    "caption": vis.caption,
                },
            )
        )
    return chunks

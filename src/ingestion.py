"""Column-aware PDF ingestion, table extraction, and visual region rendering."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request

import pdfplumber
import pypdfium2 as pdfium

from .models import ExtractedVisual, PageRecord

CHAPTER_RANGES = [
    (33, 62, "Overview"),
    (63, 78, "Part 1 / Chapter 1: Economic Growth"),
    (79, 83, "Spotlight: Middle-Income Histories"),
    (85, 99, "Chapter 2: Middle-Income Transitions"),
    (100, 115, "Chapter 3: Structural Stasis"),
    (117, 138, "Chapter 4: Creation"),
    (139, 165, "Chapter 5: Preservation"),
    (167, 190, "Chapter 6: Energy Transition"),
    (191, 215, "Chapter 7: Disciplining Incumbency"),
    (221, 251, "Chapter 8: Rewarding Merit"),
    (253, 273, "Chapter 9: Capitalizing on Crises"),
]


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFKC), unwrap line hyphens, and preserve paragraphs."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    return "\n\n".join(" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip())


def _words_to_lines(words: list[dict], y_tol: float = 4.0) -> list[str]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"] / y_tol), w["x0"]))
    lines: list[list[dict]] = []
    for w in ordered:
        if not lines or abs(w["top"] - lines[-1][0]["top"]) > y_tol:
            lines.append([w])
        else:
            lines[-1].append(w)
    return [" ".join(w["text"] for w in sorted(line, key=lambda x: x["x0"])) for line in lines]


def extract_words_reading_order(words: list[dict], page_width: float) -> str:
    """Reconstruct natural reading order for mixed single-column and two-column pages."""
    if not words:
        return ""
    mid = page_width * 0.5
    left_band = page_width * 0.46
    right_band = page_width * 0.54

    # Group into rough horizontal bands by y coordinate to isolate full-width sections vs 2-column bands.
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    bands: list[list[dict]] = []
    for w in sorted_words:
        if not bands or abs(w["top"] - bands[-1][-1]["top"]) > 18.0:
            bands.append([w])
        else:
            bands[-1].append(w)

    # Check if two-column layout applies inside a band: words on both sides, none crossing center gutter.
    output_blocks: list[str] = []
    col_left: list[dict] = []
    col_right: list[dict] = []

    def flush_columns() -> None:
        if col_left:
            output_blocks.append("\n".join(_words_to_lines(col_left)))
            col_left.clear()
        if col_right:
            output_blocks.append("\n".join(_words_to_lines(col_right)))
            col_right.clear()

    for band in bands:
        crosses_center = any(w["x0"] < left_band and w["x1"] > right_band for w in band)
        has_left = any(w["x1"] <= mid for w in band)
        has_right = any(w["x0"] >= mid for w in band)
        if not crosses_center and has_left and has_right:
            for w in band:
                if (w["x0"] + w["x1"]) * 0.5 < mid:
                    col_left.append(w)
                else:
                    col_right.append(w)
        elif col_left or col_right:
            # Continue current two-column stream if all words clearly sit inside one column.
            all_left = all(w["x1"] <= right_band for w in band)
            all_right = all(w["x0"] >= left_band for w in band)
            if not crosses_center and (all_left or all_right):
                target = col_left if all_left else col_right
                target.extend(band)
            else:
                flush_columns()
                output_blocks.append("\n".join(_words_to_lines(band)))
        else:
            output_blocks.append("\n".join(_words_to_lines(band)))
    flush_columns()
    return "\n\n".join(b for b in output_blocks if b.strip())


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Convert pdfplumber table cells into normalized Markdown pipe format."""
    cleaned: list[list[str]] = []
    for row in rows:
        vals = [
            normalize_text(re.sub(r"-\s*\n\s*", "-", cell or "").replace("\n", " ")).replace("\n\n", " ")
            for cell in row
        ]
        if any(vals):
            cleaned.append(vals)
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    padded = [r + [""] * (width - len(r)) for r in cleaned]
    header = "| " + " | ".join(padded[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, sep, *body])


def classify_page(page_number: int, text: str, in_references_block: bool) -> tuple[str, bool]:
    """Classify each physical page transparently as front_matter, empty, references, back_matter, or content."""
    if page_number <= 32:
        return "front_matter", False
    if page_number >= 274:
        return "back_matter", False
    stripped = text.strip()
    if not stripped:
        return "empty", False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    head = " ".join(lines[:3])
    has_visual = bool(re.search(r"\b(Table|Figure|Map|Box)\s+(?:[A-Z]+\d*\.|\d+\.)+\d+", stripped))
    has_section_start = bool(re.search(r"\b(Chapter\s+\d+|Part\s+\d+|Spotlight|Key messages|Overview)\b", head))
    if any(re.fullmatch(r"(?:\d+\s*\|\s*)?References", line) for line in lines[:8]) and not has_visual:
        return "references", True
    if in_references_block:
        if has_section_start or has_visual:
            return "content", False
        return "references", True
    return "content", False


def detect_visual_captions(text: str) -> list[tuple[str, str]]:
    """Extract numbered Table, Figure, Map, and Box captions from normalized page text."""
    results: list[tuple[str, str]] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(
            r"^(Table|TABLE|Figure|FIGURE|Map|MAP|Box|BOX)\s+((?:[A-Z]+\d*\.|\d+\.)+\d+[a-z]?)\b(.*)",
            s,
        )
        if m:
            kind = m.group(1).lower()
            if kind in ("table", "figure", "map", "box"):
                results.append((kind, s))
    return results


def _chapter_for_page(page_number: int) -> str:
    for start, end, label in CHAPTER_RANGES:
        if start <= page_number <= end:
            return label
    return "WDR 2024"


def _printed_page_from_text(text: str, page_number: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:2] + lines[-2:]:
        m = re.search(r"(?:^|\|\s*)(\d{1,3})(?:\s*\||$)", line)
        if m:
            return m.group(1)
    return str(max(1, page_number - 32))


def download_if_missing(url: str, path: Path, expected_sha256: str = "") -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS source URLs are allowed")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Day07-Student-Lab/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {digest}")
    return digest


def build_corpus(root: Path, render_images: bool = True, force_render: bool = False, render_scale: float = 2.2) -> tuple[list[PageRecord], list[ExtractedVisual], dict]:
    """Ingest all physical pages of WDR_2024.pdf, preserving tables, visual assets, and page logs."""
    config = json.loads((root / "configs/experiment.json").read_text(encoding="utf-8"))
    corpus_cfg = config["corpus"]
    pdf_path = root / corpus_cfg["pdf_path"]
    sha256 = download_if_missing(corpus_cfg["source_url"], pdf_path, corpus_cfg.get("sha256", ""))

    crops_dir = root / "data/crops"
    if render_images:
        crops_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = pdfium.PdfDocument(str(pdf_path)) if render_images else None
    pages_out: list[PageRecord] = []
    visuals_out: list[ExtractedVisual] = []
    prep_log: list[dict] = []
    in_refs = False
    counts: dict[str, int] = {}

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, 1):
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False) or []
            raw_col_text = extract_words_reading_order(words, float(page.width))
            # Ensure visual captions start a clean paragraph block before single-newline folding.
            raw_col_text = re.sub(
                r"\n(?=(?:Table|TABLE|Figure|FIGURE|Map|MAP|Box|BOX)\s+(?:[A-Z]+\d*\.|\d+\.)+\d+)",
                "\n\n",
                raw_col_text,
            )
            captions = detect_visual_captions(raw_col_text)
            col_text = normalize_text(raw_col_text)
            category, in_refs = classify_page(pno, raw_col_text, in_refs)
            counts[category] = counts.get(category, 0) + 1

            raw_tables = page.extract_tables() or []
            tables_md = [md for md in (table_to_markdown(t) for t in raw_tables) if md]

            entry = {
                "page": pno,
                "category": category,
                "characters": len(col_text),
                "tables_extracted": len(tables_md),
                "visual_captions": len(captions),
            }
            prep_log.append(entry)
            if category != "content":
                continue

            chapter = _chapter_for_page(pno)
            printed_page = _printed_page_from_text(col_text, pno)
            img_rel = f"data/crops/page_{pno:03d}.png"
            if render_images and pdf_doc is not None:
                img_file = root / img_rel
                if force_render or not img_file.exists():
                    pdf_doc[pno - 1].render(scale=render_scale).to_pil().save(img_file)

            pages_out.append(
                PageRecord(
                    source_id=corpus_cfg["source_id"],
                    source=corpus_cfg["source_url"],
                    page=pno,
                    printed_page=printed_page,
                    chapter=chapter,
                    text=col_text,
                    tables_markdown=tables_md,
                    visual_captions=[cap for _, cap in captions],
                    image_path=img_rel,
                    metadata={"printed_page": printed_page, "chapter": chapter, "image_path": img_rel},
                )
            )

            for idx, (vtype, cap) in enumerate(captions, 1):
                if vtype in ("figure", "map", "table"):
                    visuals_out.append(
                        ExtractedVisual(
                            visual_id=f"wdr2024-p{pno:03d}-{vtype}-{idx}",
                            visual_type=vtype,
                            source_id=corpus_cfg["source_id"],
                            page=pno,
                            printed_page=printed_page,
                            chapter=chapter,
                            caption=cap,
                            text_summary=col_text[:700],
                            image_path=img_rel,
                            bbox=[0.0, 0.0, float(page.width), float(page.height)],
                        )
                    )

    if len(pages_out) < 100:
        raise ValueError(f"Expected >= 100 content pages for advanced challenge, got {len(pages_out)}")

    manifest = {
        "source_id": corpus_cfg["source_id"],
        "title": corpus_cfg["title"],
        "source_url": corpus_cfg["source_url"],
        "handle_url": corpus_cfg["handle_url"],
        "license": corpus_cfg["license"],
        "sha256": sha256,
        "physical_pages": total_pages,
        "indexed_content_pages": len(pages_out),
        "category_counts": counts,
        "extracted_tables": sum(len(p.tables_markdown) for p in pages_out),
        "extracted_visuals": len(visuals_out),
    }

    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "data/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "results/preparation.json").write_text(json.dumps(prep_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "data/processed/pages.jsonl").write_text(
        "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in pages_out), encoding="utf-8"
    )
    (root / "data/processed/visuals.json").write_text(
        json.dumps([asdict(v) for v in visuals_out], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return pages_out, visuals_out, manifest

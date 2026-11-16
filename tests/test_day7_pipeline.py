"""TDD test suite for Day 07 Advanced Multimodal RAG pipeline."""
import pytest

from src.models import ChunkRecord, ExtractedVisual, PageRecord


def test_normalize_preserves_paragraphs_and_unwraps_hyphens():
    from src.ingestion import normalize_text

    raw = "  Middle-\nincome ﬁrms\n\nSecond   para\tline. "
    assert normalize_text(raw) == "Middleincome firms\n\nSecond para line."


def test_column_reading_order_splits_left_before_right():
    from src.ingestion import extract_words_reading_order

    # Page width 600: full-width title at top=40, then left col x=50..260, right col x=320..540
    words = [
        {"text": "Title", "x0": 50, "x1": 100, "top": 40, "bottom": 55},
        {"text": "Header", "x0": 270, "x1": 330, "top": 40, "bottom": 55},
        {"text": "Left1", "x0": 60, "x1": 110, "top": 120, "bottom": 135},
        {"text": "Right1", "x0": 330, "x1": 390, "top": 120, "bottom": 135},
        {"text": "Left2", "x0": 60, "x1": 110, "top": 150, "bottom": 165},
        {"text": "Right2", "x0": 330, "x1": 390, "top": 150, "bottom": 165},
    ]
    out = extract_words_reading_order(words, page_width=600.0)
    assert out.index("Title Header") < out.index("Left1")
    assert out.index("Left2") < out.index("Right1")
    assert out.index("Right1") < out.index("Right2")


def test_table_to_markdown_formats_cells_cleanly():
    from src.ingestion import detect_visual_captions, table_to_markdown

    rows = [["GROUP", "POP (%)"], ["Low-\nincome", "8.9"], ["High-income", "15.7"]]
    md = table_to_markdown(rows)
    assert "| GROUP | POP (%) |" in md
    assert "| Low-income | 8.9 |" in md
    assert "| High-income | 15.7 |" in md

    caps = detect_visual_captions("Table O.2 To achieve...\nFigure O.4 Middle-income...\nTable P1.1 World Bank...\nMap 6.1 In 2022...")
    assert [c[0] for c in caps] == ["table", "figure", "table", "map"]


def test_classify_page_tracks_multiline_references_block():
    from src.ingestion import classify_page

    assert classify_page(10, "Contents", False) == ("front_matter", False)
    assert classify_page(84, "", False) == ("empty", False)
    assert classify_page(215, "183 | WORLD DEVELOPMENT REPORT 2024\nNotes\n1. Foo", False) == ("content", False)
    assert classify_page(216, "184 | WORLD DEVELOPMENT REPORT 2024\nReferences\nAghion 1992", False) == ("references", True)
    assert classify_page(217, "185 | WORLD DEVELOPMENT REPORT 2024\nBarro 2010", True) == ("references", True)
    assert classify_page(79, "47 | Spotlight\nWhat can histories teach?", True) == ("content", False)
    assert classify_page(221, "Chapter 8: Rewarding Merit\nKey messages", True) == ("content", False)


def test_chunk_pages_creates_text_table_and_visual_chunks():
    from src.chunking import chunk_pages

    page = PageRecord(
        source_id="worldbank-wdr-2024",
        source="https://example.org/wdr2024.pdf",
        page=39,
        printed_page="7",
        chapter="Overview",
        text="First paragraph explains 1i, 2i, 3i transitions.\n\nSecond paragraph compares Korea.",
        tables_markdown=["| GROUP | INVESTMENT |\n| --- | --- |\n| Low-income | Higher priority |"],
        visual_captions=["Figure O.4 Middle-income countries must engineer two successive transitions"],
        image_path="data/crops/page_039.png",
    )
    visual = ExtractedVisual(
        visual_id="wdr2024-p039-fig-1",
        visual_type="figure",
        source_id="worldbank-wdr-2024",
        page=39,
        printed_page="7",
        chapter="Overview",
        caption="Figure O.4 Middle-income countries must engineer two successive transitions",
        text_summary="Relative contribution to growth vs proximity to frontier (Capital vs Productivity).",
        image_path="data/crops/page_039.png",
    )
    chunks = chunk_pages([page], [visual], max_chars=200)
    types = {c.metadata["chunk_type"] for c in chunks}
    assert {"text", "table", "visual"} <= types
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.metadata["image_path"] == "data/crops/page_039.png" for c in chunks)


def test_bm25_and_hybrid_rrf_and_budget_packing():
    from src.retrieval import BM25Retriever, HybridRRFRetriever, pack_context

    chunks = [
        ChunkRecord("c1", "wdr", "https://example.org", 39, "Table O.2 investment infusion innovation dials.", {"chunk_type": "table", "chapter": "Overview", "image_path": "p39.png"}),
        ChunkRecord("c2", "wdr", "https://example.org", 173, "Map 6.1 online job postings low-carbon technologies middle-income.", {"chunk_type": "visual", "chapter": "Chapter 6", "image_path": "p173.png"}),
        ChunkRecord("c3", "wdr", "https://example.org", 63, "Table P1.1 global GDP share 60.8 percent high-income.", {"chunk_type": "table", "chapter": "Part 1", "image_path": "p63.png"}),
    ]
    bm25 = BM25Retriever(chunks)
    hits = bm25.search("low-carbon job postings Map 6.1", top_k=2)
    assert hits and hits[0]["id"] == "c2"

    # Fake dense retriever returning scores biased to c2 then c1
    class FakeDense:
        def search(self, query, *, filters=None, top_k=5):
            pool = chunks
            if filters and "chunk_type" in filters:
                pool = [c for c in pool if c.metadata["chunk_type"] == filters["chunk_type"]]
            return [{**c.to_day7_dict(), "score": 0.9 - i * 0.1, "strategy": "dense"} for i, c in enumerate(pool[:top_k])]

    hybrid = HybridRRFRetriever(FakeDense(), bm25)
    table_hits = hybrid.search("global GDP Table P1.1", filters={"chunk_type": "table"}, top_k=2)
    assert all(h["metadata"]["chunk_type"] == "table" for h in table_hits)

    ctx, ids, images = pack_context(hits + hits, budget=220)
    assert len(ids) == len(set(ids))
    assert len(ctx) <= 220
    assert "p173.png" in images

    from src.retrieval import tokenize
    assert "o.4" in tokenize("Figure O.4 shows 3i")
    assert "p1.1" in tokenize("Table P1.1 data")


def test_multimodal_agent_supports_visual_inspection_and_abstains_on_invalid():
    from src.agent import run_multimodal_agent

    class FakeController:
        def __init__(self, steps):
            self.steps = list(steps)

        def decide(self, question, history, images):
            return self.steps.pop(0)

    class FakeRetriever:
        def search(self, query, *, filters=None, top_k=5):
            return [{"id": "c-map61", "source_id": "wdr", "source": "u", "page": 173,
                     "text": "Map 6.1 one-third of low-carbon job postings in middle-income countries.",
                     "metadata": {"chunk_type": "visual", "image_path": "p173.png"}, "score": 0.85}]

    ctrl = FakeController([
        {"action": "search", "query": "Map 6.1 low-carbon job postings"},
        {"action": "inspect_image", "page": 173},
        {"action": "answer", "answer": "One-third were in middle-income countries [c-map61].", "citations": ["c-map61"]},
    ])
    res = run_multimodal_agent("Where were low-carbon job postings?", ctrl, FakeRetriever(),
                               image_loader=lambda p: f"loaded:{p}")
    assert res["status"] == "answered"
    assert res["citations"] == ["c-map61"]
    assert any(e["event"] == "inspect_image" for e in res["events"])

    bad_ctrl = FakeController([
        {"action": "search", "query": "unknown"},
        {"action": "answer", "answer": "Made up [nonexistent].", "citations": ["nonexistent"]},
        {"action": "answer", "answer": "Still wrong [nonexistent].", "citations": ["nonexistent"]},
    ])
    res_bad = run_multimodal_agent("Missing fact?", bad_ctrl, FakeRetriever())
    assert res_bad["status"] == "abstain"


def test_vertex_gemini_controller_builds_vision_payload_and_redacts_key(monkeypatch):
    import json
    from src.agent import VertexGeminiVisionController

    captured = {}

    class FakeResp:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(req, timeout=60):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp({
            "candidates": [{"content": {"parts": [{"text": '{"action":"answer","answer":"Ok [c1].","citations":["c1"]}'}]}}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 18},
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ctrl = VertexGeminiVisionController(model="gemini-2.5-flash", api_key="SECRET_TEST_KEY")
    decision = ctrl.decide("Question?", [{"tool": "search"}], {39: "AAAA"})
    assert decision["action"] == "answer"
    assert ctrl.usage == {"prompt_tokens": 120, "completion_tokens": 18, "calls": 1}
    parts = captured["body"]["contents"][0]["parts"]
    assert any(p.get("inlineData", {}).get("mimeType") == "image/png" for p in parts)
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_agent_keeps_latest_search_chunks_when_budget_is_tight():
    from src.agent import run_multimodal_agent

    class TwoSearchCtrl:
        def __init__(self):
            self.step = 0

        def decide(self, question, history, images):
            self.step += 1
            if self.step == 1:
                return {"action": "search", "query": "first query"}
            if self.step == 2:
                return {"action": "search", "query": "second query"}
            return {"action": "answer", "answer": "Answer uses second [c2].", "citations": ["c2"]}

    class TwoSearchRet:
        def search(self, query, *, filters=None, top_k=5):
            if "first" in query:
                return [{"id": "c1", "source_id": "wdr", "source": "u", "page": 1, "text": "A" * 160, "metadata": {"chunk_type": "text"}, "score": 0.9}]
            return [{"id": "c2", "source_id": "wdr", "source": "u", "page": 2, "text": "B" * 160, "metadata": {"chunk_type": "text"}, "score": 0.8}]

    res = run_multimodal_agent("Q?", TwoSearchCtrl(), TwoSearchRet(), budget=220)
    assert res["status"] == "answered"
    assert res["citations"] == ["c2"]


def test_expand_with_neighbors_includes_same_page_text_for_visual_hits():
    from src.retrieval import expand_with_neighbors

    all_chunks = [
        ChunkRecord("vis-p67", "wdr", "u", 67, "[FIGURE] Figure 1.1", {"chunk_type": "visual", "image_path": "p67.png"}),
        ChunkRecord("txt-p67-1", "wdr", "u", 67, "Axis numbers", {"chunk_type": "text", "image_path": "p67.png"}),
        ChunkRecord("txt-p67-2", "wdr", "u", 67, "Viet Nam plans transition to high-income status by 2045.", {"chunk_type": "text", "image_path": "p67.png"}),
    ]
    top_hit = {**all_chunks[0].to_day7_dict(), "score": 0.9, "strategy": "hybrid_rerank"}
    expanded = expand_with_neighbors([top_hit], all_chunks, max_total=5)
    assert any(c["id"] == "txt-p67-2" for c in expanded)


def test_agent_requires_visual_inspection_before_answering_visual_chunk():
    from src.agent import run_multimodal_agent

    class UninspectedVisualCtrl:
        def __init__(self):
            self.step = 0
            self.saw_image_on_retry = False

        def decide(self, question, history, images):
            self.step += 1
            if self.step == 1:
                return {"action": "search", "query": "Figure 6.3 clean energy"}
            if self.step == 2:
                # Premature answer citing visual chunk before inspect_image
                return {"action": "answer", "answer": "Guess 82 [vis-175].", "citations": ["vis-175"]}
            self.saw_image_on_retry = 175 in images
            return {"action": "answer", "answer": "Verified 36 [vis-175].", "citations": ["vis-175"]}

    class VisRet:
        def search(self, query, *, filters=None, top_k=5):
            return [{
                "id": "vis-175",
                "source_id": "wdr",
                "source": "u",
                "page": 175,
                "text": "[FIGURE] Figure 6.3 clean energy financing",
                "metadata": {"chunk_type": "visual", "image_path": "data/crops/page_175.png"},
                "score": 0.9,
            }]

    ctrl = UninspectedVisualCtrl()
    res = run_multimodal_agent(
        "What is Figure 6.3 value?",
        ctrl,
        VisRet(),
        image_loader=lambda p, region="full": f"img:{p}:{region}",
    )
    assert res["status"] == "answered"
    assert "36" in res["answer"]
    assert ctrl.saw_image_on_retry is True
    assert any(e["event"] == "visual_inspect_required_hint" for e in res["events"])


def test_agent_inspect_image_supports_region_crop(tmp_path):
    from PIL import Image
    from src.agent import load_page_image_b64, run_multimodal_agent
    import base64, io

    img_path = tmp_path / "page_182.png"
    Image.new("RGB", (400, 600), color=(255, 255, 255)).save(img_path)
    full_b64 = load_page_image_b64(img_path, region="full")
    top_b64 = load_page_image_b64(img_path, region="top")
    im_full = Image.open(io.BytesIO(base64.b64decode(full_b64)))
    im_top = Image.open(io.BytesIO(base64.b64decode(top_b64)))
    assert im_full.size == (400, 600)
    assert im_top.size[1] < 600

    class RegionCtrl:
        def __init__(self):
            self.steps = [
                {"action": "search", "query": "Map 6.2"},
                {"action": "inspect_image", "page": 182, "region": "bottom_left"},
                {"action": "answer", "answer": "Done [vis-182].", "citations": ["vis-182"]},
            ]

        def decide(self, question, history, images):
            return self.steps.pop(0)

    class MapRet:
        def search(self, query, *, filters=None, top_k=5):
            return [{
                "id": "vis-182",
                "source_id": "wdr",
                "source": "u",
                "page": 182,
                "text": "[MAP] Map 6.2",
                "metadata": {"chunk_type": "visual", "image_path": str(img_path)},
                "score": 0.9,
            }]

    res = run_multimodal_agent(
        "Check inset map",
        RegionCtrl(),
        MapRet(),
        image_loader=lambda p, region="full": load_page_image_b64(p, region=region),
    )
    assert res["status"] == "answered"


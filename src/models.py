"""Shared data models for Day 07 Retrieval & Multimodal Context Engineering."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedVisual:
    visual_id: str
    visual_type: str  # "table" | "figure" | "map" | "box"
    source_id: str
    page: int
    printed_page: str
    chapter: str
    caption: str
    text_summary: str
    image_path: str
    bbox: list[float] = field(default_factory=list)


@dataclass
class PageRecord:
    source_id: str
    source: str
    page: int
    printed_page: str
    chapter: str
    text: str
    tables_markdown: list[str] = field(default_factory=list)
    visual_captions: list[str] = field(default_factory=list)
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    id: str
    source_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_day7_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "metadata": self.metadata,
        }

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    resume_id: str
    pdf_path: Path
    docx_path: Path | None = None
    ground_truth: dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"
    layout_tags: list[str] = field(default_factory=list)
    source: str = ""
    domain: str = ""
    schema_version: str = "resume_v1"

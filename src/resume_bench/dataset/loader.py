from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resume_bench.dataset.models import TestCase
from resume_bench.settings import settings

DATASET_REPO = "careerflow/resume-extract-bench"


def download_dataset(split: str = "all", revision: str = "main") -> Path:
    """Download dataset from HuggingFace to local cache."""
    from huggingface_hub import snapshot_download

    local_dir = settings.data_dir / "dataset"

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
    )

    return local_dir


def load_split(split: str, data_dir: Path | None = None) -> list[TestCase]:
    """Load a dataset split from local directory."""
    base = data_dir or settings.data_dir / "dataset"
    jsonl_path = base / f"{split}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Split '{split}' not found at {jsonl_path}. "
            f"Run 'resume-bench download' first."
        )

    cases = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            files = record.get("files", {})
            pdf_rel = files.get("pdf", "")
            docx_rel = files.get("docx")

            pdf_path = base / pdf_rel if pdf_rel else Path("")
            docx_path = base / docx_rel if docx_rel else None

            cases.append(TestCase(
                resume_id=record["resume_id"],
                pdf_path=pdf_path,
                docx_path=docx_path,
                ground_truth=record.get("ground_truth", {}),
                difficulty=record.get("difficulty", "medium"),
                layout_tags=record.get("layout_tags", []),
                source=record.get("source", ""),
                domain=record.get("domain", ""),
                schema_version=record.get("schema_version", "resume_v1"),
            ))

    return cases


def get_dataset_status() -> dict[str, Any]:
    """Check which splits are available locally."""
    base = settings.data_dir / "dataset"
    info = {}

    for split_name in ["mini", "dev", "test"]:
        jsonl_path = base / f"{split_name}.jsonl"

        if jsonl_path.exists():
            count = sum(1 for line in open(jsonl_path) if line.strip())
        else:
            count = 0

        info[split_name] = {
            "count": count,
            "path": str(jsonl_path) if jsonl_path.exists() else "not downloaded",
        }

    return info

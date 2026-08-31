from __future__ import annotations

import json
from pathlib import Path

from resume_bench.schema import get_schema


def validate_dataset(dataset_path: Path) -> list[str]:
    """Validate a JSONL dataset file against the resume schema."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema package not installed"]

    schema = get_schema()
    errors = []

    with open(dataset_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                continue

            if "resume_id" not in record:
                errors.append(f"Line {i}: missing resume_id")

            if "ground_truth" not in record:
                errors.append(f"Line {i}: missing ground_truth")
                continue

            gt = record["ground_truth"]

            try:
                jsonschema.validate(gt, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"Line {i} ({record.get('resume_id', '?')}): {e.message}")

    return errors

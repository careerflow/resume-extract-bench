from __future__ import annotations

import hashlib
import json


def cache_key(pipeline_name: str, resume_id: str, schema: dict, prompt: str) -> str:
    """Generate a cache key from pipeline, resume, schema, and prompt."""
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]

    return f"{pipeline_name}_{resume_id}_{schema_hash}_{prompt_hash}"

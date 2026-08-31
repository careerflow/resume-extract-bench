from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resume_bench.providers.base import (
    ExtractionRequest,
    Provider,
    ProviderConfigError,
    ProviderError,
)
from resume_bench.providers.registry import register_provider


@register_provider("precomputed")
class PrecomputedProvider(Provider):
    """Load pre-existing extraction results from disk."""

    def healthcheck(self) -> None:
        results_dir = self.spec.config.get("results_dir", "")

        if not results_dir:
            raise ProviderConfigError("precomputed provider requires config.results_dir")

        if not Path(results_dir).exists():
            raise ProviderConfigError(f"results_dir not found: {results_dir}")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        results_dir = Path(self.spec.config.get("results_dir", ""))

        candidates = [
            results_dir / f"{req.resume_id}.json",
            results_dir / f"{req.resume_id}.result.json",
        ]

        for path in candidates:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                return {"parsed": data, "source_path": str(path)}

        raise ProviderError(f"No precomputed result for {req.resume_id} in {results_dir}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = raw.get("parsed", raw)

        if "raw_output" in parsed:
            return parsed["raw_output"]

        return parsed

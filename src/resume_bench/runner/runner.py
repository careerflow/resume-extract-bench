from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.dataset.models import TestCase
from resume_bench.providers._pdf import pdf_to_text
from resume_bench.providers.base import (
    ExtractionRequest,
    PipelineSpec,
    ProviderError,
    RunRecord,
)
from resume_bench.providers.pipelines import PIPELINES
from resume_bench.providers.registry import create_provider
from resume_bench.schema import get_schema
from resume_bench.settings import settings


def _get_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "extraction_v1.txt"
    return prompt_path.read_text()


def _output_dir(pipeline_name: str, split: str) -> Path:
    out = settings.output_dir / pipeline_name / split
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_single(
    provider: Any,
    spec: PipelineSpec,
    case: TestCase,
    schema: dict,
    prompt: str,
    use_cache: bool,
    out_dir: Path,
) -> RunRecord:
    """Run extraction on a single resume."""
    result_path = out_dir / f"{case.resume_id}.result.json"

    if use_cache and result_path.exists():
        return RunRecord(
            resume_id=case.resume_id,
            pipeline_name=spec.pipeline_name,
            cached=True,
        )

    text = None
    if spec.input_mode.value == "text":
        text = pdf_to_text(case.pdf_path)

    req = ExtractionRequest(
        resume_id=case.resume_id,
        pdf_path=case.pdf_path,
        text=text,
        extraction_schema=schema,
        system_prompt=prompt,
    )

    started = datetime.now(timezone.utc)
    start_ms = time.monotonic_ns() // 1_000_000

    try:
        raw = provider.extract(req)
        latency = (time.monotonic_ns() // 1_000_000) - start_ms

        record = RunRecord(
            resume_id=case.resume_id,
            pipeline_name=spec.pipeline_name,
            raw_output=raw,
            latency_ms=latency,
            started_at=started,
            cost_usd=provider.estimate_cost(raw),
        )

        with open(result_path, "w") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2, default=str)

        return record

    except ProviderError as e:
        latency = (time.monotonic_ns() // 1_000_000) - start_ms

        record = RunRecord(
            resume_id=case.resume_id,
            pipeline_name=spec.pipeline_name,
            error=str(e),
            error_class=type(e).__name__,
            latency_ms=latency,
            started_at=started,
        )

        with open(result_path, "w") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2, default=str)

        return record


def run_pipelines(
    pipeline_names: list[str],
    split: str = "test",
    limit: int | None = None,
    concurrency: int = 4,
    use_cache: bool = True,
) -> dict[str, dict[str, int]]:
    """Run one or more pipelines on a dataset split."""
    cases = load_split(split)

    if limit:
        cases = cases[:limit]

    schema = get_schema()
    prompt = _get_prompt()

    specs_by_name = {s.pipeline_name: s for s in PIPELINES}
    all_stats = {}

    for name in pipeline_names:
        spec = specs_by_name.get(name)
        if not spec:
            raise ValueError(f"Unknown pipeline: {name}. Available: {sorted(specs_by_name)}")

        provider = create_provider(spec)
        out_dir = _output_dir(name, split)

        stats = {"extracted": 0, "cached": 0, "errors": 0}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _run_single, provider, spec, case, schema, prompt, use_cache, out_dir
                ): case
                for case in cases
            }

            for future in as_completed(futures):
                record = future.result()

                if record.cached:
                    stats["cached"] += 1
                elif record.error:
                    stats["errors"] += 1
                else:
                    stats["extracted"] += 1

        all_stats[name] = stats

    return all_stats

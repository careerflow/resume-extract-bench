from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.dataset.models import TestCase
from resume_bench.providers._pdf import get_document_info, pdf_to_text
from resume_bench.providers.base import (
    ExtractionRequest,
    PipelineSpec,
    ProviderError,
    RunRecord,
)
from resume_bench.providers.base import Provider as _ProviderType
from resume_bench.providers.pipelines import PIPELINES
from resume_bench.providers.registry import create_provider
from resume_bench.prompts.extraction_v1 import EXTRACTION_SYSTEM_PROMPT
from resume_bench.schema import get_schema
from resume_bench.settings import settings


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

    # Capture document metrics before extraction
    doc_info = None
    try:
        doc_info = get_document_info(case.pdf_path)
    except Exception:
        pass  # non-fatal — extraction proceeds without doc info

    text = None
    parse_meta = None
    if spec.input_mode.value == "text":
        if spec.parse_source == "llamaparse":
            from resume_bench.providers._pdf import pdf_to_markdown_llamaparse
            text, parse_meta = pdf_to_markdown_llamaparse(case.pdf_path)
        else:
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
        canonical = provider.to_canonical(raw)
        latency = (time.monotonic_ns() // 1_000_000) - start_ms

        cost_usd = provider.estimate_cost(raw)
        usage = provider.get_usage(raw)
        retry_count = raw.get("retry_count", 0) if isinstance(raw, dict) else 0

        # Compute cost per page
        cost_per_page = None
        if cost_usd is not None and doc_info and doc_info.page_count:
            cost_per_page = cost_usd / doc_info.page_count

        # Determine input method from spec or raw response
        input_method = None
        if isinstance(raw, dict):
            input_method = raw.get("input_method")
        if input_method is None:
            if spec.input_mode.value == "pdf":
                input_method = "pdf_native"
            elif spec.input_mode.value == "images":
                input_method = "rasterized_images"
            elif spec.parse_source == "llamaparse":
                input_method = "llamaparse_text"
            else:
                input_method = "pymupdf_text"

        parse_cost_usd = None
        parse_source_val = spec.parse_source
        if parse_meta:
            parse_cost_usd = parse_meta.get("parse_cost_usd")
            if parse_source_val is None:
                parse_source_val = "llamaparse"

        record = RunRecord(
            resume_id=case.resume_id,
            pipeline_name=spec.pipeline_name,
            raw_output=raw,
            output=canonical,
            latency_ms=latency,
            started_at=started,
            cost_usd=cost_usd,
            document=doc_info,
            usage=usage,
            retry_count=retry_count,
            cost_per_page_usd=cost_per_page,
            input_method=input_method,
            parse_cost_usd=parse_cost_usd,
            parse_source=parse_source_val,
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
            document=doc_info,
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
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

    cases = load_split(split)

    if limit:
        cases = cases[:limit]

    schema = get_schema()
    prompt = EXTRACTION_SYSTEM_PROMPT

    specs_by_name = {s.pipeline_name: s for s in PIPELINES}
    all_stats = {}

    for name in pipeline_names:
        spec = specs_by_name.get(name)
        if not spec:
            raise ValueError(f"Unknown pipeline: {name}. Available: {sorted(specs_by_name)}")

        provider = create_provider(spec)
        out_dir = _output_dir(name, split)

        stats = {"extracted": 0, "cached": 0, "errors": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[status]}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"{name}", total=len(cases), status="starting..."
            )

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        _run_single, provider, spec, case, schema, prompt, use_cache, out_dir
                    ): case
                    for case in cases
                }

                for future in as_completed(futures):
                    record = future.result()
                    case = futures[future]

                    if record.cached:
                        stats["cached"] += 1
                        label = "cached"
                    elif record.error:
                        stats["errors"] += 1
                        label = "error"
                    else:
                        stats["extracted"] += 1
                        label = "ok"

                    done = stats["extracted"] + stats["cached"] + stats["errors"]
                    progress.update(
                        task, advance=1,
                        status=f"{done}/{len(cases)} | {stats['extracted']} ok, {stats['cached']} cached, {stats['errors']} err",
                    )

        all_stats[name] = stats

    return all_stats

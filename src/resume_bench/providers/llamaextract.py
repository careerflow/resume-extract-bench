from __future__ import annotations

from typing import Any

from resume_bench.providers.base import (
    ExtractionRequest,
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderTransientError,
    ProviderUsage,
)
from resume_bench.providers.registry import register_provider

# Credit-based pricing (source: ExtractBench / LlamaIndex pricing page)
_USD_PER_CREDIT = 1.25 / 1000  # $0.00125 per credit

_EXTRACT_CREDITS_PER_PAGE: dict[str, int] = {
    "cost_effective": 5,
    "agentic": 15,
    "agentic_plus": 50,
}
_PARSE_CREDITS_PER_PAGE: dict[str, int] = {
    "fast": 1,
    "cost_effective": 3,
    "agentic": 10,
    "agentic_plus": 45,
}
# Which parse tier each extract tier actually runs
_DEFAULT_PARSE_TIER: dict[str, str] = {
    "agentic_plus": "agentic",  # agentic_plus extract runs agentic parse
}

# Legacy flat per-page rates (fallback)
_LLAMAEXTRACT_COST_PER_PAGE: dict[str, float] = {
    "agentic_plus": 0.056,
    "agentic": 0.042,
    "cost_effective": 0.028,
    "turbo": 0.014,
}


def _llamaextract_cost(tier: str, pages: float) -> float:
    """Compute LlamaExtract cost using the two-leg credit model."""
    extract_credits = _EXTRACT_CREDITS_PER_PAGE.get(tier)
    parse_tier = _DEFAULT_PARSE_TIER.get(tier, tier)
    parse_credits = _PARSE_CREDITS_PER_PAGE.get(parse_tier)
    if extract_credits is not None and parse_credits is not None:
        total_credits = (extract_credits + parse_credits) * pages
        return total_credits * _USD_PER_CREDIT
    # Fallback to legacy flat rate
    rate = _LLAMAEXTRACT_COST_PER_PAGE.get(tier, 0.056)
    return pages * rate


@register_provider("llamaextract")
class LlamaExtractProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.llama_cloud_api_key:
            raise ProviderConfigError("RESUME_BENCH_LLAMA_CLOUD_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from llama_cloud import LlamaCloud

        from resume_bench.settings import settings

        client = LlamaCloud(api_key=settings.llama_cloud_api_key)
        tier = self.spec.config.get("tier", "agentic_plus")
        version = self.spec.config.get("version")

        prompt = req.system_prompt or ""

        try:
            with open(req.pdf_path, "rb") as f:
                upload_response = client.files.create(file=f, purpose="extract")

            file_id = upload_response.id

            config: dict[str, Any] = {
                "data_schema": req.extraction_schema,
                "extraction_target": "per_doc",
                "tier": tier,
                "confidence_scores": True,
                "system_prompt": prompt,
            }
            if version:
                config["version"] = version

            job = client.extract.run(
                file_input=file_id,
                configuration=config,
            )

            data = job.extract_result if job.extract_result else {}

            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif hasattr(data, "dict"):
                data = data.dict()
            elif not isinstance(data, dict):
                import json

                data = json.loads(str(data)) if data else {}

            # Capture usage metadata from job
            pages_billed = None
            pages_extracted = None
            job_id = getattr(job, "id", None)
            field_confidence: dict[str, Any] | None = None

            job_meta = getattr(job, "metadata", None)
            if job_meta:
                usage_obj = getattr(job_meta, "usage", None)
                if usage_obj:
                    pages_billed = getattr(usage_obj, "num_pages_billed", None)
                    pages_extracted = getattr(usage_obj, "num_pages_extracted", None)

            # Capture per-field confidence from extract_metadata
            extract_meta = getattr(job, "extract_metadata", None)
            if extract_meta:
                fm = getattr(extract_meta, "field_metadata", None)
                if fm:
                    if hasattr(fm, "model_dump"):
                        field_confidence = fm.model_dump()
                    elif isinstance(fm, dict):
                        field_confidence = fm

            return {
                "parsed": data,
                "tier": tier,
                "version": version,
                "job_id": str(job_id) if job_id else None,
                "pages_billed": pages_billed,
                "pages_extracted": pages_extracted,
                "field_confidence": field_confidence,
            }

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"LlamaExtract rate limit: {e}")
            raise ProviderError(f"LlamaExtract failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        tier = raw.get("tier", "agentic_plus")
        pages = raw.get("pages_billed") or 2  # fallback to 2 pages
        return _llamaextract_cost(tier, pages)

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        return ProviderUsage(
            pages_billed=raw.get("pages_billed"),
            pages_extracted=raw.get("pages_extracted"),
            request_id=raw.get("job_id"),
            model_id=raw.get("tier"),
        )

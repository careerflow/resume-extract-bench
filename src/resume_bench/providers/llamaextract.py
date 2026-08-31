from __future__ import annotations

from typing import Any

from resume_bench.providers.base import (
    ExtractionRequest,
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderTransientError,
)
from resume_bench.providers.registry import register_provider


@register_provider("llamaextract")
class LlamaExtractProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.llama_cloud_api_key:
            raise ProviderConfigError("RESUME_BENCH_LLAMA_CLOUD_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from llama_cloud import LlamaCloud

        from resume_bench.settings import settings

        client = LlamaCloud(token=settings.llama_cloud_api_key)
        tier = self.spec.config.get("tier", "agentic")

        try:
            extraction = client.extraction.create_extraction(
                schema=req.extraction_schema,
                input_files=[str(req.pdf_path)],
                config={"tier": tier},
            )

            if extraction.results and len(extraction.results) > 0:
                result = extraction.results[0]
                return {
                    "parsed": result.data if hasattr(result, "data") else {},
                    "tier": tier,
                    "extraction_id": extraction.id if hasattr(extraction, "id") else None,
                }

            return {"parsed": {}, "tier": tier, "error": "no results returned"}

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"LlamaExtract rate limit: {e}")
            raise ProviderError(f"LlamaExtract failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

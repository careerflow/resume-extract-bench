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


@register_provider("reducto")
class ReductoProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.reducto_api_key:
            raise ProviderConfigError("RESUME_BENCH_REDUCTO_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from reducto import Reducto

        from resume_bench.settings import settings

        client = Reducto(api_key=settings.reducto_api_key)

        try:
            result = client.extract(
                document_url=str(req.pdf_path),
                schema=req.extraction_schema,
            )

            parsed = result.result if hasattr(result, "result") else {}
            if isinstance(parsed, str):
                import json
                parsed = json.loads(parsed)

            return {"parsed": parsed}

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"Reducto rate limit: {e}")
            raise ProviderError(f"Reducto extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

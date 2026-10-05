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
            with open(req.pdf_path, "rb") as f:
                upload_response = client.upload(file=f)

            file_id = upload_response.file_id

            result = client.extract.run(
                input={"type": "file_id", "file_id": file_id},
                instructions={"schema": req.extraction_schema},
            )

            if hasattr(result, "result") and result.result:
                extraction = (
                    result.result[0]
                    if isinstance(result.result, list)
                    else result.result
                )

                if hasattr(extraction, "content"):
                    data = extraction.content
                elif hasattr(extraction, "model_dump"):
                    data = extraction.model_dump()
                else:
                    data = extraction
            else:
                data = result

            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif hasattr(data, "dict"):
                data = data.dict()
            elif not isinstance(data, dict):
                import json

                data = json.loads(str(data))

            # Capture usage metadata from result
            credits_used = None
            num_pages = None
            job_id = getattr(result, "job_id", None)
            duration_s = getattr(result, "duration", None)
            confidence_val = getattr(result, "confidence", None)
            confidence_reason = getattr(result, "confidence_reason", None)

            usage_obj = getattr(result, "usage", None)
            if usage_obj:
                credits_used = getattr(usage_obj, "credits", None)
                num_pages = getattr(usage_obj, "num_pages", None)

            return {
                "parsed": data,
                "job_id": str(job_id) if job_id else None,
                "credits_used": credits_used,
                "pages_extracted": num_pages,
                "duration_s": duration_s,
                "confidence": str(confidence_val) if confidence_val is not None else None,
                "confidence_reason": str(confidence_reason) if confidence_reason else None,
            }

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"Reducto rate limit: {e}")
            raise ProviderError(f"Reducto extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        # Convert server-side duration (seconds) to milliseconds
        duration_s = raw.get("duration_s")
        server_ms = int(duration_s * 1000) if duration_s is not None else None

        # Reducto returns confidence as "high"/"low" string; map to 0.0-1.0
        conf_str = raw.get("confidence")
        confidence = None
        if conf_str == "high":
            confidence = 1.0
        elif conf_str == "low":
            confidence = 0.0

        return ProviderUsage(
            credits_used=raw.get("credits_used"),
            pages_extracted=raw.get("pages_extracted"),
            request_id=raw.get("job_id"),
            server_processing_ms=server_ms,
            confidence=confidence,
            confidence_reason=raw.get("confidence_reason"),
        )

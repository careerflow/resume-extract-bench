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

            return {"parsed": data}

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"Reducto rate limit: {e}")
            raise ProviderError(f"Reducto extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

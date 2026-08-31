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

        client = LlamaCloud(api_key=settings.llama_cloud_api_key)
        tier = self.spec.config.get("tier", "agentic_plus")

        try:
            with open(req.pdf_path, "rb") as f:
                upload_response = client.files.create(file=f, purpose="extract")

            file_id = upload_response.id

            job = client.extract.run(
                file_input=file_id,
                configuration={
                    "data_schema": req.extraction_schema,
                    "tier": tier,
                    "confidence_scores": True,
                    "system_prompt": req.system_prompt,
                },
            )

            data = job.extract_result if job.extract_result else {}

            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif hasattr(data, "dict"):
                data = data.dict()
            elif not isinstance(data, dict):
                import json

                data = json.loads(str(data)) if data else {}

            return {"parsed": data, "tier": tier}

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"LlamaExtract rate limit: {e}")
            raise ProviderError(f"LlamaExtract failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

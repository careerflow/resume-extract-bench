from __future__ import annotations

import json
from typing import Any

from resume_bench.providers._llm_common import parse_json_response
from resume_bench.providers.base import (
    ExtractionRequest,
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderTransientError,
)
from resume_bench.providers.registry import register_provider


@register_provider("google")
class GoogleProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.google_api_key:
            raise ProviderConfigError("RESUME_BENCH_GOOGLE_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from google import genai

        from resume_bench.settings import settings

        client = genai.Client(api_key=settings.google_api_key)
        model = self.spec.config.get("model", "gemini-2.5-pro")

        prompt = (
            f"{req.system_prompt}\n\n"
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
            f"Resume text:\n{req.text}\n\n"
            f"Respond with only valid JSON matching the schema."
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            content = response.text or "{}"

            usage_meta = response.usage_metadata
            usage = {}
            if usage_meta:
                usage = {
                    "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage_meta, "candidates_token_count", 0),
                }

            return {
                "parsed": parse_json_response(content),
                "model": model,
                "usage": usage,
            }

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                raise ProviderTransientError(f"Google rate limit: {e}")
            raise ProviderError(f"Google extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        model = raw.get("model", "")

        rates = {
            "gemini-2.5-pro": (1.25, 10.00),
            "gemini-2.5-flash": (0.15, 0.60),
        }
        input_rate, output_rate = rates.get(model, (1.25, 10.00))

        return (prompt * input_rate + completion * output_rate) / 1_000_000

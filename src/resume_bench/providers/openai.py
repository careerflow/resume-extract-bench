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


@register_provider("openai")
class OpenAIProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.openai_api_key:
            raise ProviderConfigError("RESUME_BENCH_OPENAI_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from openai import OpenAI

        from resume_bench.settings import settings

        client = OpenAI(api_key=settings.openai_api_key)
        model = self.spec.config.get("model", "gpt-4o")

        messages = [
            {"role": "system", "content": req.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Extract structured data from this resume according to the schema.\n\n"
                    f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
                    f"Resume text:\n{req.text}"
                ),
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"

            return {
                "parsed": parse_json_response(content),
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise ProviderTransientError(f"OpenAI rate limit: {e}")
            raise ProviderError(f"OpenAI extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        model = raw.get("model", "gpt-4o")

        rates = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4.1": (2.00, 8.00),
            "gpt-5.6-sol": (5.00, 20.00),
        }
        input_rate, output_rate = rates.get(model, (2.50, 10.00))

        return (prompt * input_rate + completion * output_rate) / 1_000_000

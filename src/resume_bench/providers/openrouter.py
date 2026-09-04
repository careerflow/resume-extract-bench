from __future__ import annotations

import time
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


@register_provider("openrouter")
class OpenRouterProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.openrouter_api_key:
            raise ProviderConfigError("RESUME_BENCH_OPENROUTER_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        import json

        from openai import OpenAI

        from resume_bench.settings import settings

        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = self.spec.config.get("model", "")

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

        max_retries = 5
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=32768,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                if content is None:
                    # Some thinking models consume all tokens on reasoning
                    content = "{}"

                return {
                    "parsed": parse_json_response(content),
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                }

            except Exception as e:
                last_error = e
                error_str = str(e)
                if "402" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        wait = min(30 * (2 ** attempt), 300)
                        time.sleep(wait)
                        continue
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    raise ProviderTransientError(f"OpenRouter rate limit: {e}")
                raise ProviderError(f"OpenRouter extraction failed: {e}")

        raise ProviderError(f"OpenRouter extraction failed after {max_retries} retries: {last_error}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        # OpenRouter pricing varies by model; return None to skip cost estimation
        return None

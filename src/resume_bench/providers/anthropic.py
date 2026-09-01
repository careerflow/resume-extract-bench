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


@register_provider("anthropic")
class AnthropicProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.anthropic_api_key:
            raise ProviderConfigError("RESUME_BENCH_ANTHROPIC_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        import anthropic

        from resume_bench.settings import settings

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = self.spec.config.get("model", "claude-sonnet-5")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=req.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Extract structured data from this resume according to the schema.\n\n"
                            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
                            f"Resume text:\n{req.text}\n\n"
                            f"Respond with only valid JSON matching the schema."
                        ),
                    },
                ],
            )

            text_blocks = [b for b in response.content if b.type == "text"]
            content = text_blocks[0].text if text_blocks else "{}"

            return {
                "parsed": parse_json_response(content),
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except anthropic.RateLimitError as e:
            raise ProviderTransientError(f"Anthropic rate limit: {e}")
        except Exception as e:
            raise ProviderError(f"Anthropic extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage", {})
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)

        model = raw.get("model", "")

        rates = {
            "claude-sonnet-5": (3.00, 15.00),
            "claude-opus-5": (15.00, 75.00),
            "claude-haiku-4-5-20251001": (0.80, 4.00),
        }
        input_rate, output_rate = rates.get(model, (3.00, 15.00))

        return (input_tok * input_rate + output_tok * output_rate) / 1_000_000

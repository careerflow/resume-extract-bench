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
    ProviderUsage,
)
from resume_bench.providers.registry import register_provider


@register_provider("google")
class GoogleProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.google_api_key:
            raise ProviderConfigError("RESUME_BENCH_GOOGLE_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        if req.text is not None:
            return self._extract_text(req)
        return self._extract_pdf(req)

    def _extract_text(self, req: ExtractionRequest) -> dict[str, Any]:
        """Text-mode extraction: send extracted text in the prompt."""
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

        return self._call_gemini(client, model, prompt)

    def _extract_pdf(self, req: ExtractionRequest) -> dict[str, Any]:
        """PDF-direct extraction: send PDF as inline bytes via Gemini API."""
        from google import genai
        from google.genai import types

        from resume_bench.settings import settings

        client = genai.Client(api_key=settings.google_api_key)
        model = self.spec.config.get("model", "gemini-2.5-pro")

        prompt = (
            f"{req.system_prompt}\n\n"
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
            f"Respond with only valid JSON matching the schema."
        )

        with open(req.pdf_path, "rb") as f:
            pdf_bytes = f.read()

        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        return self._call_gemini(client, model, [pdf_part, prompt])

    def _call_gemini(
        self,
        client: Any,
        model: str,
        contents: Any,
    ) -> dict[str, Any]:
        """Shared Gemini API call logic for both text and PDF modes."""
        from google import genai

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            content = response.text or "{}"

            usage_meta = response.usage_metadata
            usage: dict[str, Any] = {}
            if usage_meta:
                usage = {
                    "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage_meta, "candidates_token_count", 0),
                }
                thoughts = getattr(usage_meta, "thoughts_token_count", None)
                if thoughts is not None:
                    usage["reasoning_tokens"] = thoughts
                total = getattr(usage_meta, "total_token_count", None)
                if total is not None:
                    usage["total_tokens"] = total
                cached = getattr(usage_meta, "cached_content_token_count", None)
                if cached is not None:
                    usage["cached_input_tokens"] = cached

            # Capture finish_reason from first candidate
            finish_reason = None
            candidates = getattr(response, "candidates", None)
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                if finish_reason is not None:
                    finish_reason = str(finish_reason)

            return {
                "parsed": parse_json_response(content),
                "model": model,
                "usage": usage,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                raise ProviderTransientError(f"Google rate limit: {e}")
            raise ProviderError(f"Google extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    # USD per million tokens (input, output).  Uses longest-prefix matching.
    _PRICING_PER_M: dict[str, tuple[float, float]] = {
        "gemini-3.8-flash": (0.75, 3.75),     # introductory through 2026-12-31; standard: 1.50/7.50
        "gemini-3.6-flash": (0.75, 3.75),     # introductory through 2026-12-31; standard: 1.50/7.50
        "gemini-3.5-flash": (1.50, 9.00),
        "gemini-3.1-flash-lite": (0.25, 1.50),
        "gemini-3.1-pro": (2.00, 12.00),
        "gemini-2.5-flash": (0.15, 0.60),
        "gemini-2.5-pro": (1.25, 10.00),
    }

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        model = raw.get("model", "")

        # Longest-prefix match for dated model IDs
        matches = [(p, r) for p, r in self._PRICING_PER_M.items() if model.startswith(p)]
        input_rate, output_rate = (
            max(matches, key=lambda x: len(x[0]))[1] if matches else (1.25, 10.00)
        )

        return (prompt * input_rate + completion * output_rate) / 1_000_000

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        usage = raw.get("usage", {})
        input_tok = usage.get("prompt_tokens")
        output_tok = usage.get("completion_tokens")
        return ProviderUsage(
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=usage.get("reasoning_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            finish_reason=raw.get("finish_reason"),
            model_id=raw.get("model"),
        )

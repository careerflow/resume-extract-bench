from __future__ import annotations

import base64
import time
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

# Per-million token rates: (input, output).
# Covers all models from the internal benchmark + ExtractBench leaderboard.
_OPENROUTER_RATES: dict[str, tuple[float, float]] = {
    # Qwen family
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "qwen/qwen3.8-flash-next": (0.15, 0.47),
    "qwen/qwen3.8-27b": (0.42, 3.00),
    "qwen/qwen3.6-35b-a3b": (0.15, 1.00),
    "qwen/qwen3.5-35b-a3b": (0.15, 1.00),
    "qwen/qwen3.5-9b": (0.10, 0.15),
    "qwen/qwen3-235b-a22b": (0.20, 0.60),
    "qwen/qwen3-30b-a3b": (0.13, 0.13),
    "qwen/qwen2.5-72b-instruct": (0.36, 0.40),
    "qwen/qwen2.5-7b-instruct": (0.05, 0.10),
    # Moonshot / Kimi
    "moonshotai/kimi-k3": (2.70, 13.50),
    "moonshotai/kimi-k2": (0.60, 2.50),
    "moonshotai/kimi-vl-a3b-thinking": (0.0, 0.0),
    # Z-AI / GLM
    "z-ai/glm-5.3-flash": (0.15, 0.50),
    "thudm/glm-4-32b": (0.10, 0.10),
    # Google (via OpenRouter)
    "google/gemini-3.8-flash": (0.75, 3.75),    # introductory through 2026-12-31
    "google/gemini-3.5-flash": (1.50, 9.00),
    "google/gemma-4-26b-a4b-it": (0.0675, 0.225),
    "google/gemma-4-e4b-it": (0.04, 0.22),
    "google/gemma-3-27b-it": (0.10, 0.20),
    # OpenGVLab / OpenBMB
    "opengvlab/internvl3-14b": (0.10, 0.50),
    "openbmb/minicpm-v-4.5": (0.66, 1.11),
    # DeepSeek
    "deepseek/deepseek-v4.1-flash": (0.015, 0.677),
}


@register_provider("openrouter")
class OpenRouterProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.openrouter_api_key:
            raise ProviderConfigError("RESUME_BENCH_OPENROUTER_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from resume_bench.providers.base import InputMode

        if self.spec.input_mode == InputMode.IMAGES:
            return self._extract_images(req)
        if req.text is not None:
            return self._extract_text(req)
        return self._extract_pdf(req)

    def _extract_text(self, req: ExtractionRequest) -> dict[str, Any]:
        """Text-mode extraction: send extracted text in the message."""
        import json

        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
            f"Resume text:\n{req.text}"
        )

        messages = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_openrouter(messages, req)

    def _extract_pdf(self, req: ExtractionRequest) -> dict[str, Any]:
        """PDF-direct extraction: send PDF as inline base64 data URI."""
        import json

        with open(req.pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("ascii")

        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```"
        )

        user_content = [
            {
                "type": "file",
                "file": {
                    "filename": "resume.pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            },
            {"type": "text", "text": prompt},
        ]

        messages = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self._call_openrouter(messages, req)

    def _extract_images(self, req: ExtractionRequest) -> dict[str, Any]:
        """Vision-mode extraction: rasterize PDF pages to PNG and send as image_url blocks."""
        import json

        from resume_bench.providers._pdf import pdf_to_base64_images

        images_b64 = pdf_to_base64_images(req.pdf_path)

        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```"
        )

        user_content: list[dict] = []
        for img_b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self._call_openrouter(messages, req)
        result["input_method"] = "rasterized_images"
        result["page_count"] = len(images_b64)
        return result

    def _call_openrouter(
        self,
        messages: list[dict],
        req: ExtractionRequest,
    ) -> dict[str, Any]:
        """Shared OpenRouter API call logic with retry handling."""
        from openai import OpenAI

        from resume_bench.settings import settings

        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = self.spec.config.get("model", "")

        max_retries = 5
        last_error: Exception | None = None
        retry_count = 0

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

                finish_reason = getattr(response.choices[0], "finish_reason", None)
                response_id = getattr(response, "id", None)

                return {
                    "parsed": parse_json_response(content),
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                    "response_id": response_id,
                    "finish_reason": finish_reason,
                    "retry_count": retry_count,
                }

            except Exception as e:
                last_error = e
                error_str = str(e)
                if "402" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        retry_count += 1
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
        model = raw.get("model", "")
        # Exact match first, then longest-prefix
        rates = _OPENROUTER_RATES.get(model)
        if rates is None:
            matches = [(p, r) for p, r in _OPENROUTER_RATES.items() if model.startswith(p)]
            if not matches:
                return None
            rates = max(matches, key=lambda x: len(x[0]))[1]
        usage = raw.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        input_rate, output_rate = rates
        return (prompt * input_rate + completion * output_rate) / 1_000_000

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        usage = raw.get("usage", {})
        input_tok = usage.get("prompt_tokens")
        output_tok = usage.get("completion_tokens")
        return ProviderUsage(
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=(input_tok or 0) + (output_tok or 0) or None,
            request_id=raw.get("response_id"),
            finish_reason=raw.get("finish_reason"),
            model_id=raw.get("model"),
        )

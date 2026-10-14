from __future__ import annotations

import base64
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


@register_provider("anthropic")
class AnthropicProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.anthropic_api_key:
            raise ProviderConfigError("RESUME_BENCH_ANTHROPIC_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        if req.text is not None:
            return self._extract_text(req)
        return self._extract_pdf(req)

    def _extract_text(self, req: ExtractionRequest) -> dict[str, Any]:
        """Text-mode extraction: send extracted text in the message."""
        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```\n\n"
            f"Resume text:\n{req.text}"
        )
        user_content: str | list = prompt
        return self._call_anthropic(req, user_content)

    def _extract_pdf(self, req: ExtractionRequest) -> dict[str, Any]:
        """PDF-direct extraction: send PDF as inline base64 document block."""
        with open(req.pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("ascii")

        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```"
        )

        user_content: list = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        return self._call_anthropic(req, user_content)

    def _call_anthropic(
        self,
        req: ExtractionRequest,
        user_content: str | list,
    ) -> dict[str, Any]:
        """Shared Anthropic API call logic for both text and PDF modes."""
        import anthropic

        from resume_bench.settings import settings

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = self.spec.config.get("model", "claude-sonnet-5")

        tool_def = {
            "name": "extract_resume",
            "description": "Extract structured resume data",
            "input_schema": req.extraction_schema,
        }

        create_kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=8192,
            system=req.system_prompt,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "extract_resume"},
            messages=[{"role": "user", "content": user_content}],
        )
        # claude-opus-4-8 and newer models deprecate the temperature parameter
        if "4-8" not in model and "5" not in model:
            create_kwargs["temperature"] = 0

        try:
            response = client.messages.create(**create_kwargs)

            # Build common usage + metadata dict
            usage_dict = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
            cache_read = getattr(response.usage, "cache_read_input_tokens", None)
            if cache_creation is not None:
                usage_dict["cache_creation_tokens"] = cache_creation
            if cache_read is not None:
                usage_dict["cache_read_tokens"] = cache_read

            common = {
                "model": model,
                "usage": usage_dict,
                "response_id": getattr(response, "id", None),
                "finish_reason": getattr(response, "stop_reason", None),
            }

            # Primary path: extract the structured tool_use input
            for block in response.content:
                if block.type == "tool_use":
                    parsed = block.input if isinstance(block.input, dict) else {}
                    return {"parsed": parsed, **common}

            # Fallback: if no tool_use block, try raw text content
            for block in response.content:
                if block.type == "text":
                    return {"parsed": parse_json_response(block.text), **common}

            return {"parsed": {}, **common}

        except anthropic.RateLimitError as e:
            raise ProviderTransientError(f"Anthropic rate limit: {e}")
        except Exception as e:
            raise ProviderError(f"Anthropic extraction failed: {e}")

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    # USD per million tokens (input, output).  Uses longest-prefix matching.
    _PRICING_PER_M: dict[str, tuple[float, float]] = {
        "claude-opus-4-8": (5.00, 25.00),
        "claude-opus-5": (5.00, 25.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-sonnet-5": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
    }

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage", {})
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)

        model = raw.get("model", "")

        # Longest-prefix match for dated model IDs
        matches = [(p, r) for p, r in self._PRICING_PER_M.items() if model.startswith(p)]
        input_rate, output_rate = (
            max(matches, key=lambda x: len(x[0]))[1] if matches else (3.00, 15.00)
        )

        return (input_tok * input_rate + output_tok * output_rate) / 1_000_000

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        usage = raw.get("usage", {})
        input_tok = usage.get("input_tokens")
        output_tok = usage.get("output_tokens")
        total = (input_tok or 0) + (output_tok or 0) or None
        return ProviderUsage(
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=total,
            cache_creation_tokens=usage.get("cache_creation_tokens"),
            cache_read_tokens=usage.get("cache_read_tokens"),
            request_id=raw.get("response_id"),
            finish_reason=raw.get("finish_reason"),
            model_id=raw.get("model"),
        )

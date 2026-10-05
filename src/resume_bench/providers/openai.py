from __future__ import annotations

import copy
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

# Model prefixes that support Structured Outputs (strict JSON Schema mode).
_STRUCTURED_OUTPUT_PREFIXES = ("gpt-6", "gpt-5.6", "gpt-5.5", "gpt-5.4")

# USD per million tokens (input, output).  Uses longest-prefix matching.
_OPENAI_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
}


def _make_strict_schema(schema: dict) -> dict:
    """Convert a JSON Schema to OpenAI Structured Outputs strict format.

    Recursively adds ``additionalProperties: false`` and ``required`` to
    every object node, as required by OpenAI's strict JSON Schema mode.
    """
    schema = copy.deepcopy(schema)

    def _fix(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            for prop in node["properties"].values():
                _fix(prop)
        if node.get("type") == "array" and "items" in node:
            _fix(node["items"])
        # Recurse into anyOf branches (e.g. nullable objects)
        if "anyOf" in node:
            for branch in node["anyOf"]:
                _fix(branch)
        return node

    return _fix(schema)


@register_provider("openai")
class OpenAIProvider(Provider):

    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.openai_api_key:
            raise ProviderConfigError("RESUME_BENCH_OPENAI_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        if req.text is not None:
            return self._extract_text(req)
        return self._extract_pdf(req)

    def _extract_text(self, req: ExtractionRequest) -> dict[str, Any]:
        """Text-mode extraction: send extracted text in the message."""
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

        return self._call_openai(client, model, messages, req)

    def _extract_pdf(self, req: ExtractionRequest) -> dict[str, Any]:
        """PDF-direct extraction: upload PDF via Files API, reference by file_id."""
        from openai import OpenAI

        from resume_bench.settings import settings

        client = OpenAI(api_key=settings.openai_api_key)
        model = self.spec.config.get("model", "gpt-4o")

        prompt = (
            f"Extract structured data from this resume according to the schema.\n\n"
            f"Schema:\n```json\n{json.dumps(req.extraction_schema, indent=2)}\n```"
        )

        # Upload PDF to Files API
        with open(req.pdf_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="user_data")

        try:
            content = [
                {"type": "file", "file": {"file_id": file_obj.id}},
                {"type": "text", "text": prompt},
            ]

            messages = [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": content},
            ]

            return self._call_openai(client, model, messages, req)
        finally:
            try:
                client.files.delete(file_obj.id)
            except Exception:
                pass

    def _call_openai(
        self,
        client: Any,
        model: str,
        messages: list[dict],
        req: ExtractionRequest,
    ) -> dict[str, Any]:
        """Shared OpenAI API call logic for both text and PDF modes."""
        use_structured = any(model.startswith(p) for p in _STRUCTURED_OUTPUT_PREFIXES)

        try:
            if use_structured:
                strict_schema = _make_strict_schema(req.extraction_schema)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "resume_extraction",
                            "strict": True,
                            "schema": strict_schema,
                        },
                    },
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )

            content = response.choices[0].message.content or "{}"
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            response_id = getattr(response, "id", None)
            system_fingerprint = getattr(response, "system_fingerprint", None)

            usage_dict: dict[str, Any] = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }

            if response.usage:
                # Capture reasoning tokens from completion_tokens_details (GPT-5.x)
                comp_details = getattr(response.usage, "completion_tokens_details", None)
                if comp_details:
                    reasoning = getattr(comp_details, "reasoning_tokens", None)
                    if reasoning is not None:
                        usage_dict["reasoning_tokens"] = reasoning

                # Capture cached prompt tokens from prompt_tokens_details
                prompt_details = getattr(response.usage, "prompt_tokens_details", None)
                if prompt_details:
                    cached = getattr(prompt_details, "cached_tokens", None)
                    if cached is not None:
                        usage_dict["cached_input_tokens"] = cached

            return {
                "parsed": parse_json_response(content),
                "model": model,
                "usage": usage_dict,
                "response_id": response_id,
                "finish_reason": finish_reason,
                "system_fingerprint": system_fingerprint,
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

        # Longest-prefix match for dated model IDs
        matches = [(p, r) for p, r in _OPENAI_PRICING_PER_M.items() if model.startswith(p)]
        input_rate, output_rate = (
            max(matches, key=lambda x: len(x[0]))[1] if matches else (2.50, 10.00)
        )

        return (prompt * input_rate + completion * output_rate) / 1_000_000

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        usage = raw.get("usage", {})
        return ProviderUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=(usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0) or None,
            reasoning_tokens=usage.get("reasoning_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            request_id=raw.get("response_id"),
            finish_reason=raw.get("finish_reason"),
            model_id=raw.get("model"),
            system_fingerprint=raw.get("system_fingerprint"),
        )

from __future__ import annotations

import json
import re


def parse_json_response(text: str) -> dict:
    """Parse an LLM response as JSON, stripping markdown fences if present."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

        return {}

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

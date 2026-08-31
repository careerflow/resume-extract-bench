"""Resume extraction schema module.

This module provides access to the JSON Schema definition for resume extraction.
The schema is loaded from resume_v1.json and exposed through the RESUME_SCHEMA
constant and get_schema() function.
"""

import json
from importlib.resources import files
from typing import Any


def _load_schema() -> dict[str, Any]:
    """Load the resume schema from the JSON file.

    Returns:
        dict: The parsed JSON schema definition.
    """
    schema_path = files(__package__) / "resume_v1.json"
    schema_text = schema_path.read_text(encoding="utf-8")

    return json.loads(schema_text)


RESUME_SCHEMA: dict[str, Any] = _load_schema()


def get_schema() -> dict[str, Any]:
    """Get the resume extraction schema.

    Returns:
        dict: A copy of the JSON Schema definition for resume extraction.
    """
    return RESUME_SCHEMA.copy()

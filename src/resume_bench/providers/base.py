from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class InputMode(str, Enum):
    PDF = "pdf"
    TEXT = "text"


class PipelineSpec(BaseModel):
    pipeline_name: str
    provider_name: str
    input_mode: InputMode
    config: dict[str, Any] = {}
    per_file_timeout_s: float = 600
    notes: str = ""


class ExtractionRequest(BaseModel):
    resume_id: str
    pdf_path: Path
    text: str | None = None
    extraction_schema: dict[str, Any]
    system_prompt: str

    model_config = {"arbitrary_types_allowed": True}


class RunRecord(BaseModel):
    resume_id: str
    pipeline_name: str
    raw_output: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    error_class: str | None = None
    latency_ms: int = 0
    cost_usd: float | None = None
    started_at: datetime | None = None
    cached: bool = False

    model_config = {"arbitrary_types_allowed": True}


class ProviderError(Exception):
    pass

class ProviderTransientError(ProviderError):
    pass

class ProviderPermanentError(ProviderError):
    pass

class ProviderConfigError(ProviderError):
    pass

class ProviderTimeoutError(ProviderError):
    pass


class Provider(ABC):

    def __init__(self, spec: PipelineSpec, settings: Any = None):
        self.spec = spec
        self.settings = settings

    @abstractmethod
    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        """Call the extraction system. Return raw JSON dict. Raise ProviderError on failure."""

    @abstractmethod
    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map raw provider output to the canonical resume schema."""

    def estimate_cost(self, raw: dict[str, Any]) -> float | None:
        return None

    def healthcheck(self) -> None:
        """Raise ProviderConfigError if the provider is misconfigured."""
        pass

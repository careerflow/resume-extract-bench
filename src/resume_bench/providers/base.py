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
    IMAGES = "images"


class PipelineSpec(BaseModel):
    pipeline_name: str
    provider_name: str
    input_mode: InputMode
    config: dict[str, Any] = {}
    per_file_timeout_s: float = 600
    notes: str = ""
    parse_source: str | None = None  # "llamaparse", "pymupdf", or None


class DocumentInfo(BaseModel):
    """Document metrics captured before extraction."""

    page_count: int | None = None
    file_size_bytes: int | None = None
    char_count: int | None = None
    word_count: int | None = None
    token_count_estimate: int | None = None  # char_count // 4


class ProviderUsage(BaseModel):
    """Structured usage details from the provider API response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None  # OpenAI reasoning, Google thoughts
    cache_creation_tokens: int | None = None  # Anthropic
    cache_read_tokens: int | None = None  # Anthropic
    cached_input_tokens: int | None = None  # OpenAI prompt cache hits
    credits_used: float | None = None  # Reducto/Extend
    pages_billed: int | None = None  # LlamaExtract
    pages_extracted: int | None = None  # LlamaExtract
    request_id: str | None = None  # Provider job/request ID
    finish_reason: str | None = None  # stop_reason / finish_reason
    model_id: str | None = None  # Actual model used
    system_fingerprint: str | None = None  # OpenAI checkpoint identifier
    server_processing_ms: int | None = None  # Server-side processing time
    confidence: float | None = None  # Document-level extraction confidence
    confidence_reason: str | None = None  # Explanation for confidence level


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
    document: DocumentInfo | None = None
    usage: ProviderUsage | None = None
    retry_count: int = 0
    cost_per_page_usd: float | None = None
    input_method: str | None = None  # "pdf_native", "rasterized_images", "llamaparse_text"
    parse_cost_usd: float | None = None
    parse_source: str | None = None  # "llamaparse", "pymupdf", or None

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

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        """Extract structured usage info from the raw extraction response."""
        return None

    def healthcheck(self) -> None:
        """Raise ProviderConfigError if the provider is misconfigured."""
        pass

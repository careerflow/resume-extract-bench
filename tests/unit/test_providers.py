"""Tests for provider data models, serialization, and cost estimation."""
from __future__ import annotations

import json

from resume_bench.providers.base import (
    DocumentInfo,
    InputMode,
    PipelineSpec,
    ProviderUsage,
    RunRecord,
)


class TestRunRecordBackwardCompat:
    """Old JSON without new fields should load without error."""

    def test_minimal_fields(self):
        old = {"resume_id": "x", "pipeline_name": "y", "latency_ms": 100}
        record = RunRecord(**old)
        assert record.resume_id == "x"
        assert record.pipeline_name == "y"
        assert record.latency_ms == 100
        assert record.document is None
        assert record.usage is None
        assert record.retry_count == 0
        assert record.cost_per_page_usd is None

    def test_round_trip_json(self):
        old = {"resume_id": "x", "pipeline_name": "y"}
        record = RunRecord(**old)
        dumped = json.loads(json.dumps(record.model_dump(mode="json"), default=str))
        reloaded = RunRecord(**dumped)
        assert reloaded.resume_id == "x"
        assert reloaded.document is None
        assert reloaded.usage is None


class TestDocumentInfoSerialization:

    def test_round_trip(self):
        info = DocumentInfo(
            page_count=3,
            file_size_bytes=125000,
            char_count=4800,
            word_count=720,
            token_count_estimate=1200,
        )
        dumped = json.loads(json.dumps(info.model_dump(mode="json")))
        reloaded = DocumentInfo(**dumped)
        assert reloaded.page_count == 3
        assert reloaded.file_size_bytes == 125000
        assert reloaded.char_count == 4800
        assert reloaded.word_count == 720
        assert reloaded.token_count_estimate == 1200

    def test_all_none(self):
        info = DocumentInfo()
        assert info.page_count is None
        assert info.file_size_bytes is None
        assert info.word_count is None


class TestProviderUsageSerialization:

    def test_round_trip(self):
        usage = ProviderUsage(
            input_tokens=1500,
            output_tokens=800,
            total_tokens=2300,
            reasoning_tokens=100,
            request_id="req_abc123",
            finish_reason="stop",
            model_id="gpt-5.6",
        )
        dumped = json.loads(json.dumps(usage.model_dump(mode="json")))
        reloaded = ProviderUsage(**dumped)
        assert reloaded.input_tokens == 1500
        assert reloaded.output_tokens == 800
        assert reloaded.total_tokens == 2300
        assert reloaded.reasoning_tokens == 100
        assert reloaded.request_id == "req_abc123"
        assert reloaded.finish_reason == "stop"
        assert reloaded.model_id == "gpt-5.6"

    def test_all_none(self):
        usage = ProviderUsage()
        assert usage.input_tokens is None
        assert usage.credits_used is None
        assert usage.pages_billed is None
        assert usage.cached_input_tokens is None
        assert usage.system_fingerprint is None
        assert usage.server_processing_ms is None
        assert usage.confidence is None
        assert usage.confidence_reason is None

    def test_llamaextract_fields(self):
        usage = ProviderUsage(
            pages_billed=3,
            pages_extracted=3,
            request_id="job_xyz",
            model_id="agentic_plus",
        )
        assert usage.pages_billed == 3
        assert usage.input_tokens is None

    def test_reducto_fields(self):
        usage = ProviderUsage(
            credits_used=1.5,
            pages_extracted=2,
            request_id="job_123",
        )
        assert usage.credits_used == 1.5


class TestOpenAIEstimateCost:

    def test_gpt4o_rate(self):
        from resume_bench.providers.openai import OpenAIProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openai",
            input_mode=InputMode.TEXT,
            config={"model": "gpt-4o"},
        )
        provider = OpenAIProvider(spec)

        raw = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        cost = provider.estimate_cost(raw)
        # gpt-4o: input=2.50, output=10.00 per million
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_gpt56_sol_rate(self):
        from resume_bench.providers.openai import OpenAIProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openai",
            input_mode=InputMode.TEXT,
            config={"model": "gpt-5.6-sol"},
        )
        provider = OpenAIProvider(spec)

        raw = {
            "model": "gpt-5.6-sol",
            "usage": {"prompt_tokens": 2000, "completion_tokens": 1000},
        }
        cost = provider.estimate_cost(raw)
        expected = (2000 * 5.00 + 1000 * 30.00) / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10


class TestOpenRouterEstimateCost:

    def test_known_model_rate(self):
        from resume_bench.providers.openrouter import OpenRouterProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openrouter",
            input_mode=InputMode.TEXT,
            config={"model": "qwen/qwen3-235b-a22b"},
        )
        provider = OpenRouterProvider(spec)

        raw = {
            "model": "qwen/qwen3-235b-a22b",
            "usage": {"prompt_tokens": 5000, "completion_tokens": 2000},
        }
        cost = provider.estimate_cost(raw)
        # qwen3-235b: input=0.20, output=0.60 per million
        expected = (5000 * 0.20 + 2000 * 0.60) / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_none(self):
        from resume_bench.providers.openrouter import OpenRouterProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openrouter",
            input_mode=InputMode.TEXT,
            config={"model": "unknown/model"},
        )
        provider = OpenRouterProvider(spec)

        raw = {
            "model": "unknown/model",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        assert provider.estimate_cost(raw) is None


class TestLlamaExtractEstimateCost:

    def test_agentic_plus_rate(self):
        from resume_bench.providers.llamaextract import LlamaExtractProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="llamaextract",
            input_mode=InputMode.PDF,
            config={"tier": "agentic_plus"},
        )
        provider = LlamaExtractProvider(spec)

        raw = {"tier": "agentic_plus", "pages_billed": 3}
        cost = provider.estimate_cost(raw)
        assert cost is not None
        # Credit model: (50 extract + 10 parse) × 3 pages × $0.00125/credit
        assert abs(cost - (50 + 10) * 3 * 0.00125) < 1e-10

    def test_cost_effective_rate(self):
        from resume_bench.providers.llamaextract import LlamaExtractProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="llamaextract",
            input_mode=InputMode.PDF,
            config={"tier": "cost_effective"},
        )
        provider = LlamaExtractProvider(spec)

        raw = {"tier": "cost_effective", "pages_billed": 5}
        cost = provider.estimate_cost(raw)
        assert cost is not None
        # Credit model: (5 extract + 3 parse) × 5 pages × $0.00125/credit
        assert abs(cost - (5 + 3) * 5 * 0.00125) < 1e-10

    def test_fallback_pages(self):
        from resume_bench.providers.llamaextract import LlamaExtractProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="llamaextract",
            input_mode=InputMode.PDF,
            config={"tier": "agentic"},
        )
        provider = LlamaExtractProvider(spec)

        # No pages_billed → fallback to 2
        raw = {"tier": "agentic"}
        cost = provider.estimate_cost(raw)
        assert cost is not None
        # Credit model: (15 extract + 10 parse) × 2 pages × $0.00125/credit
        assert abs(cost - (15 + 10) * 2 * 0.00125) < 1e-10


class TestRunRecordWithMetadata:
    """RunRecord with all new fields populated."""

    def test_full_record_round_trip(self):
        record = RunRecord(
            resume_id="abc123",
            pipeline_name="openai_gpt56",
            latency_ms=2500,
            cost_usd=0.0075,
            document=DocumentInfo(page_count=2, file_size_bytes=80000, char_count=3200, word_count=480, token_count_estimate=800),
            usage=ProviderUsage(input_tokens=1500, output_tokens=800, total_tokens=2300, model_id="gpt-5.6"),
            retry_count=1,
            cost_per_page_usd=0.00375,
        )
        dumped = json.loads(json.dumps(record.model_dump(mode="json"), default=str))
        reloaded = RunRecord(**dumped)
        assert reloaded.document is not None
        assert reloaded.document.page_count == 2
        assert reloaded.document.word_count == 480
        assert reloaded.usage is not None
        assert reloaded.usage.input_tokens == 1500
        assert reloaded.retry_count == 1
        assert reloaded.cost_per_page_usd == 0.00375


class TestOpenAIGetUsage:

    def test_cached_tokens_and_fingerprint(self):
        from resume_bench.providers.openai import OpenAIProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openai",
            input_mode=InputMode.TEXT,
            config={"model": "gpt-5.6"},
        )
        provider = OpenAIProvider(spec)

        raw = {
            "model": "gpt-5.6",
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "reasoning_tokens": 100,
                "cached_input_tokens": 1500,
            },
            "response_id": "chatcmpl-abc",
            "finish_reason": "stop",
            "system_fingerprint": "fp_abc123",
        }
        usage = provider.get_usage(raw)
        assert usage is not None
        assert usage.cached_input_tokens == 1500
        assert usage.reasoning_tokens == 100
        assert usage.system_fingerprint == "fp_abc123"
        assert usage.finish_reason == "stop"


class TestGoogleGetUsage:

    def test_finish_reason_and_cached(self):
        from resume_bench.providers.google import GoogleProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="google",
            input_mode=InputMode.TEXT,
            config={"model": "gemini-2.5-flash"},
        )
        provider = GoogleProvider(spec)

        raw = {
            "model": "gemini-2.5-flash",
            "usage": {
                "prompt_tokens": 3000,
                "completion_tokens": 800,
                "reasoning_tokens": 200,
                "total_tokens": 4000,
                "cached_input_tokens": 1000,
            },
            "finish_reason": "STOP",
        }
        usage = provider.get_usage(raw)
        assert usage is not None
        assert usage.finish_reason == "STOP"
        assert usage.cached_input_tokens == 1000
        assert usage.reasoning_tokens == 200
        assert usage.total_tokens == 4000


class TestReductoGetUsage:

    def test_server_processing_and_confidence(self):
        from resume_bench.providers.reducto import ReductoProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="reducto",
            input_mode=InputMode.PDF,
        )
        provider = ReductoProvider(spec)

        raw = {
            "job_id": "job_xyz",
            "credits_used": 2.0,
            "pages_extracted": 3,
            "duration_s": 4.5,
            "confidence": "high",
            "confidence_reason": None,
        }
        usage = provider.get_usage(raw)
        assert usage is not None
        assert usage.server_processing_ms == 4500
        assert usage.confidence == 1.0
        assert usage.confidence_reason is None
        assert usage.credits_used == 2.0

    def test_low_confidence(self):
        from resume_bench.providers.reducto import ReductoProvider
        from resume_bench.providers.base import PipelineSpec, InputMode

        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="reducto",
            input_mode=InputMode.PDF,
        )
        provider = ReductoProvider(spec)

        raw = {
            "confidence": "low",
            "confidence_reason": "Document is heavily image-based",
            "duration_s": None,
        }
        usage = provider.get_usage(raw)
        assert usage is not None
        assert usage.confidence == 0.0
        assert usage.confidence_reason == "Document is heavily image-based"
        assert usage.server_processing_ms is None


class TestProviderUsageNewFields:

    def test_all_new_fields_round_trip(self):
        usage = ProviderUsage(
            input_tokens=2000,
            output_tokens=500,
            cached_input_tokens=1500,
            system_fingerprint="fp_abc",
            server_processing_ms=3200,
            confidence=0.95,
            confidence_reason="All fields extracted successfully",
        )
        dumped = json.loads(json.dumps(usage.model_dump(mode="json")))
        reloaded = ProviderUsage(**dumped)
        assert reloaded.cached_input_tokens == 1500
        assert reloaded.system_fingerprint == "fp_abc"
        assert reloaded.server_processing_ms == 3200
        assert reloaded.confidence == 0.95
        assert reloaded.confidence_reason == "All fields extracted successfully"


class TestInputModeImages:
    """InputMode.IMAGES enum value."""

    def test_images_value(self):
        assert InputMode.IMAGES.value == "images"

    def test_all_modes(self):
        assert set(m.value for m in InputMode) == {"pdf", "text", "images"}


class TestPipelineSpecParseSource:
    """PipelineSpec with parse_source field."""

    def test_parse_source_default_none(self):
        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openrouter",
            input_mode=InputMode.TEXT,
        )
        assert spec.parse_source is None

    def test_parse_source_llamaparse(self):
        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openrouter",
            input_mode=InputMode.TEXT,
            parse_source="llamaparse",
        )
        assert spec.parse_source == "llamaparse"

    def test_images_mode_spec(self):
        spec = PipelineSpec(
            pipeline_name="test",
            provider_name="openrouter",
            input_mode=InputMode.IMAGES,
            config={"model": "moonshotai/kimi-k3"},
        )
        assert spec.input_mode == InputMode.IMAGES
        assert spec.parse_source is None


class TestRunRecordInputMethod:
    """RunRecord with input pipeline metadata fields."""

    def test_new_fields_default_none(self):
        record = RunRecord(resume_id="x", pipeline_name="y")
        assert record.input_method is None
        assert record.parse_cost_usd is None
        assert record.parse_source is None

    def test_new_fields_round_trip(self):
        record = RunRecord(
            resume_id="abc",
            pipeline_name="openrouter_qwen38_flash",
            latency_ms=3000,
            input_method="llamaparse_text",
            parse_cost_usd=0.0037,
            parse_source="llamaparse",
        )
        dumped = json.loads(json.dumps(record.model_dump(mode="json"), default=str))
        reloaded = RunRecord(**dumped)
        assert reloaded.input_method == "llamaparse_text"
        assert reloaded.parse_cost_usd == 0.0037
        assert reloaded.parse_source == "llamaparse"

    def test_rasterized_images_method(self):
        record = RunRecord(
            resume_id="abc",
            pipeline_name="openrouter_kimi_k3",
            input_method="rasterized_images",
        )
        assert record.input_method == "rasterized_images"
        assert record.parse_source is None

    def test_backward_compat_old_json(self):
        """Old JSON without new fields should load without error."""
        old = {
            "resume_id": "x",
            "pipeline_name": "y",
            "latency_ms": 100,
            "cost_usd": 0.01,
        }
        record = RunRecord(**old)
        assert record.input_method is None
        assert record.parse_cost_usd is None
        assert record.parse_source is None


class TestPdfToBase64Images:
    """pdf_to_base64_images returns list of non-empty base64 strings."""

    def test_returns_list_of_strings(self, tmp_path):
        """Create a minimal PDF and verify rasterization."""
        import fitz

        # Create a 1-page PDF with some text
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((50, 100), "Hello World")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        from resume_bench.providers._pdf import pdf_to_base64_images

        images = pdf_to_base64_images(pdf_path)
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], str)
        assert len(images[0]) > 100  # non-trivial base64 content

    def test_multi_page(self, tmp_path):
        """Multi-page PDF returns one image per page."""
        import fitz

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=200, height=200)
            page.insert_text((50, 100), f"Page {i + 1}")
        pdf_path = tmp_path / "multi.pdf"
        doc.save(str(pdf_path))
        doc.close()

        from resume_bench.providers._pdf import pdf_to_base64_images

        images = pdf_to_base64_images(pdf_path)
        assert len(images) == 3
        for img in images:
            assert isinstance(img, str)
            assert len(img) > 0

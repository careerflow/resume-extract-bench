from resume_bench.providers.base import InputMode, PipelineSpec

PIPELINES: list[PipelineSpec] = [
    PipelineSpec(
        pipeline_name="llamaextract_agentic_plus",
        provider_name="llamaextract",
        input_mode=InputMode.PDF,
        config={"tier": "agentic_plus", "confidence_scores": True},
        per_file_timeout_s=1800,
    ),
    PipelineSpec(
        pipeline_name="llamaextract_agentic",
        provider_name="llamaextract",
        input_mode=InputMode.PDF,
        config={"tier": "agentic"},
    ),
    PipelineSpec(
        pipeline_name="llamaextract_cost_effective",
        provider_name="llamaextract",
        input_mode=InputMode.PDF,
        config={"tier": "cost_effective"},
    ),
    PipelineSpec(
        pipeline_name="llamaextract_turbo",
        provider_name="llamaextract",
        input_mode=InputMode.PDF,
        config={"tier": "turbo"},
    ),
    PipelineSpec(
        pipeline_name="reducto_extract",
        provider_name="reducto",
        input_mode=InputMode.PDF,
    ),
    PipelineSpec(
        pipeline_name="openai_gpt-4o_text",
        provider_name="openai",
        input_mode=InputMode.TEXT,
        config={"model": "gpt-4o"},
    ),
    PipelineSpec(
        pipeline_name="openai_gpt-4.1_text",
        provider_name="openai",
        input_mode=InputMode.TEXT,
        config={"model": "gpt-4.1"},
    ),
    PipelineSpec(
        pipeline_name="openai_gpt-5.6-sol_text",
        provider_name="openai",
        input_mode=InputMode.TEXT,
        config={"model": "gpt-5.6-sol"},
    ),
    PipelineSpec(
        pipeline_name="anthropic_claude-haiku-4.5_text",
        provider_name="anthropic",
        input_mode=InputMode.TEXT,
        config={"model": "claude-haiku-4-5-20251001"},
    ),
    PipelineSpec(
        pipeline_name="anthropic_claude-sonnet-5_text",
        provider_name="anthropic",
        input_mode=InputMode.TEXT,
        config={"model": "claude-sonnet-5"},
    ),
    PipelineSpec(
        pipeline_name="anthropic_claude-opus-5_text",
        provider_name="anthropic",
        input_mode=InputMode.TEXT,
        config={"model": "claude-opus-5"},
    ),
    PipelineSpec(
        pipeline_name="google_gemini-2.5-pro_text",
        provider_name="google",
        input_mode=InputMode.TEXT,
        config={"model": "gemini-2.5-pro"},
    ),
    PipelineSpec(
        pipeline_name="google_gemini-2.5-flash_text",
        provider_name="google",
        input_mode=InputMode.TEXT,
        config={"model": "gemini-2.5-flash"},
    ),
]

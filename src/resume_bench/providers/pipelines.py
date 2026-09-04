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
    # OpenRouter models
    PipelineSpec(
        pipeline_name="openrouter_qwen38_flash",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "qwen/qwen3.8-flash"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_qwen38_27b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "qwen/qwen3.8-27b"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_qwen35_35b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "qwen/qwen3.5-35b-a3b"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_qwen35_9b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "qwen/qwen3.5-9b"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_kimi_k3",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "moonshotai/kimi-k3"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_glm53_flash",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "z-ai/glm-5.3-flash"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_gemini35_flash",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "google/gemini-3.5-flash"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_gemma4_26b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "google/gemma-4-26b-a4b-it"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_internvl35_14b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "opengvlab/internvl3-14b"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_kimi_vl_a3b",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "moonshotai/kimi-vl-a3b-thinking"},
    ),
    PipelineSpec(
        pipeline_name="openrouter_minicpm_v45",
        provider_name="openrouter",
        input_mode=InputMode.TEXT,
        config={"model": "openbmb/minicpm-v-4.5"},
    ),
]

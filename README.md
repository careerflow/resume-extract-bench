# ResumeExtractBench

A benchmark for evaluating structured resume extraction systems.

## Quick Start

```bash
pip install -e ".[dev]"

# Download the dataset
resume-bench download

# Run a pipeline
resume-bench run openai_gpt-4o_text --split dev --limit 5

# Grade results
resume-bench grade openai_gpt-4o_text --split dev

# View leaderboard
resume-bench leaderboard --split dev
```

## Pipelines

| Pipeline | Provider | Input |
|----------|----------|-------|
| llamaextract_agentic_plus | LlamaExtract | PDF |
| llamaextract_agentic | LlamaExtract | PDF |
| reducto_extract | Reducto | PDF |
| openai_gpt-4o_text | OpenAI | Text |
| openai_gpt-4.1_text | OpenAI | Text |
| anthropic_claude-sonnet-4_text | Anthropic | Text |
| anthropic_claude-opus-4_text | Anthropic | Text |
| google_gemini-2.5-pro_text | Google | Text |
| google_gemini-2.5-flash_text | Google | Text |

## Metrics

- **Entity F1** - Hungarian-aligned entity matching with Jaro-Winkler similarity
- **Field Accuracy** - Per-field correctness within matched entities
- **Description Token F1** - Bag-of-words F1 for bullet-point text
- **Omission / Hallucination Rates** - Missed and spurious entity rates

## Schema

The benchmark uses a 9-section resume schema: basics, experience, education, projects, personalSummary, certifications, awards, volunteering, skills.

See `src/resume_bench/schema/resume_v1.json` for the full JSON Schema.

## License

Apache 2.0 - see [LICENSE](LICENSE).

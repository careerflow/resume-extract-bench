# ResumeExtractBench

A benchmark for evaluating structured resume extraction systems.

## Quick Start

```bash
pip install -e ".[dev]"

# Download the dataset
resume-bench download

# Run a pipeline (text-based provider)
resume-bench run openai_gpt-4o_text --split test --limit 5

# Run LlamaExtract (PDF-based provider)
resume-bench run llamaextract_agentic_plus --split test --limit 5

# Run without cache (re-extract even if results exist)
resume-bench run llamaextract_agentic_plus --split test --limit 5 --no-cache

# Grade results (default: ExtractBench-exact scoring)
resume-bench grade llamaextract_agentic_plus --split test

# Grade with fuzzy entity-level scoring
resume-bench grade llamaextract_agentic_plus --split test --fuzzy

# Grade multiple pipelines at once
resume-bench grade llamaextract_agentic_plus openai_gpt-4o_text --split test

# View leaderboard
resume-bench leaderboard --split test

# Show help
resume-bench --help
resume-bench run --help
```

## Bring Your Own Predictions

Already have extraction output? Grade it directly without writing any code:

```bash
resume-bench grade-file my_predictions.jsonl --split test
```

Each line of the JSONL file should be:

```json
{"resume_id": "022c0307-...", "prediction": {"basics": {"fname": "...", ...}, "experience": {"sectionTitle": "Work Experience", "items": [...]}, ...}}
```

> **Note:** Sections can be either wrapped objects (`"experience": {"sectionTitle": "...", "items": [...]}`) or flat arrays (`"experience": [...]`). The grader accepts both formats.

## Pipelines

| Pipeline | Provider | Input | Notes |
|----------|----------|-------|-------|
| llamaextract_agentic_plus | LlamaExtract | PDF | Highest quality tier |
| llamaextract_agentic | LlamaExtract | PDF | |
| llamaextract_cost_effective | LlamaExtract | PDF | |
| llamaextract_turbo | LlamaExtract | PDF | Fastest tier |
| reducto_extract | Reducto | PDF | |
| openai_gpt-4o_text | OpenAI | Text | |
| openai_gpt-4.1_text | OpenAI | Text | |
| openai_gpt-5.6-sol_text | OpenAI | Text | Reasoning model |
| anthropic_claude-haiku-4.5_text | Anthropic | Text | |
| anthropic_claude-sonnet-5_text | Anthropic | Text | |
| anthropic_claude-opus-5_text | Anthropic | Text | |
| google_gemini-2.5-pro_text | Google | Text | |
| google_gemini-2.5-flash_text | Google | Text | |

## Metrics

### Headline: Entity F1

The primary metric is **macro Entity F1** averaged across all non-vacuous sections.

For each section (experience, education, etc.):
1. Predicted entities are aligned to ground truth entities using the **Hungarian algorithm** (optimal bipartite matching)
2. Alignment scores use **Jaro-Winkler similarity** on key fields (company, position, institution, etc.)
3. A match requires similarity >= **0.5 threshold**
4. **Precision** = correctly matched predictions / total predictions
5. **Recall** = matched ground truth entities / total ground truth entities
6. **F1** = harmonic mean of precision and recall

### Additional Metrics

- **Basics Field Accuracy** - Per-field Jaro-Winkler accuracy on contact info (name, email, phone, location).
- **Description Quality** - Positional edit distance ratio for bullet-point text: each bullet is compared index-to-index (bullet[0] vs bullet[0], etc.) using normalized Levenshtein distance. Missing or extra bullets score 0.
- **Omission Rate** - Fraction of ground truth entities with no matching prediction (missed entities)
- **Hallucination Rate** - Fraction of predicted entities with no matching ground truth (spurious entities)
- **Cost and Latency** - Per-resume averages, surfaced in the leaderboard

### Scoring Rules

- **Failed extractions score 0** in the headline F1 (not skipped). A provider that crashes on 50% of resumes gets penalized.
- **Empty-vs-empty sections are vacuous** and excluded from the headline average. If ground truth has no experience and the prediction also has none, that section doesn't count.
- Both **penalized F1** (all resumes) and **completed-only F1** (successful extractions only) are reported.

## Schema

The benchmark uses a 14-section resume schema. Each array section is wrapped as `{sectionTitle, items}` and `personalSummary` as `{sectionTitle, text}`.

| Section | Type | In Headline F1 |
|---------|------|-----------------|
| basics | Singleton (contact info) | Yes - reported as field accuracy |
| experience | Entity list | Yes |
| education | Entity list | Yes |
| projects | Entity list | Yes |
| personalSummary | Text | Yes |
| certifications | Entity list | Yes |
| awards | Entity list | Yes |
| volunteering | Entity list | Yes |
| skills | Flat list (category + skills) | Yes |
| publications | Entity list | Yes |
| languages | Entity list | Yes |
| interests | Entity list | Yes |
| profiles | Entity list | Yes |
| customSections | Entity list | Yes |

See `src/resume_bench/schema/resume_v1.json` for the full JSON Schema.

## Adding a Custom Provider

Create a provider in ~20 lines:

```python
from resume_bench.providers.base import Provider, ExtractionRequest
from resume_bench.providers.registry import register_provider

@register_provider("my_tool")
class MyProvider(Provider):

    def healthcheck(self) -> None:
        # Check API key / connectivity
        pass

    def extract(self, req: ExtractionRequest) -> dict:
        # req.text has the resume text, req.extraction_schema has the target schema
        result = call_your_api(req.text, req.extraction_schema)
        return {"parsed": result}

    def to_canonical(self, raw: dict) -> dict:
        return raw["parsed"]
```

Then add a pipeline spec in `src/resume_bench/providers/pipelines.py`:

```python
PipelineSpec(
    pipeline_name="my_tool_v1",
    provider_name="my_tool",
    input_mode=InputMode.TEXT,
)
```

Run it:

```bash
resume-bench run my_tool_v1 --split test
resume-bench grade my_tool_v1 --split test
```

## Environment Variables

All config uses the `RESUME_BENCH_` prefix:

```
RESUME_BENCH_OPENAI_API_KEY=sk-...
RESUME_BENCH_ANTHROPIC_API_KEY=sk-ant-...
RESUME_BENCH_GOOGLE_API_KEY=...
RESUME_BENCH_LLAMA_CLOUD_API_KEY=llx-...
RESUME_BENCH_REDUCTO_API_KEY=...
```

## License

Apache 2.0 - see [LICENSE](LICENSE).

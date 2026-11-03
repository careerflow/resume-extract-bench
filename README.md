# ResumeExtractBench

A benchmark for evaluating structured resume extraction systems.

## Quick Start

```bash
git clone https://github.com/careerflow/resume-extract-bench.git
cd resume-extract-bench
pip install -e ".[dev]"

# Download the dataset
resume-bench download

# Run a pipeline (text-based provider)
resume-bench run anthropic_claude-haiku-4-5_text --split mini --limit 5

# Run LlamaExtract (PDF-based provider)
resume-bench run llamaextract_agentic_plus --split mini --limit 5

# Run without cache (re-extract even if results exist)
resume-bench run llamaextract_agentic_plus --split mini --limit 5 --no-cache

# Grade results (ResumeBench-Strict scoring, verbatim by default)
resume-bench grade llamaextract_agentic_plus --split mini

# Grade multiple pipelines at once
resume-bench grade llamaextract_agentic_plus anthropic_claude-haiku-4-5_text --split mini

# View leaderboard
resume-bench leaderboard --split mini

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

Run `resume-bench providers` to see all available pipelines. A selection:

| Pipeline | Provider | Input | Notes |
|----------|----------|-------|-------|
| llamaextract_agentic_plus | LlamaExtract | PDF | Highest quality tier |
| llamaextract_agentic | LlamaExtract | PDF | |
| llamaextract_cost_effective | LlamaExtract | PDF | |
| llamaextract_turbo | LlamaExtract | PDF | Fastest tier |
| reducto_extract | Reducto | PDF | |
| openai_gpt-4o_text | OpenAI | Text | |
| openai_gpt-4.1_text | OpenAI | Text | |
| anthropic_claude-haiku-4-5_text | Anthropic | Text | |
| anthropic_claude-sonnet-4-6_text | Anthropic | Text | |
| anthropic_claude-opus-4-8_text | Anthropic | Text | |
| google_gemini-2.5-pro_text | Google | Text | |
| google_gemini-2.5-flash_text | Google | Text | |

## Scoring: ResumeBench-Strict

The sole scoring method is **ResumeBench-Strict** — a per-field scoring algorithm with identity-gated matching.

### Rules

1. **Per-field units.** Each field in a matched entity is scored independently — partial credit for records that match some fields but not all. Description bullets are compared positionally (bullet[0] vs bullet[0], etc.).

2. **Identity matching.** Entities in structured sections (experience, education, projects, etc.) are matched by identity fields (e.g. company + position + years) using direct greedy matching with a configurable gate (default 1.0 = all identity fields must match). Keyword sections (languages, interests, profiles, publications) use Hungarian alignment.

3. **Empty is not free.** Basics fields count only where the label or the parser has a value. Skills: one unit per skill.

4. **Section-averaged.** Per document, F1 / precision / recall are computed per section and averaged, so every section weighs the same.

5. **Verbatim text (default).** Strings match only after whitespace collapsing — case and punctuation count. Typographic variants (smart quotes, em dashes, soft hyphens) are folded. Contacts and locations are compared case-insensitively. Pass `--no-verbatim` to disable.

### Scoring Rules

- **Failed extractions score 0** in the headline F1 (not skipped). A provider that crashes on 50% of resumes gets penalized.
- **Empty-vs-empty sections are vacuous** and excluded from the headline average.
- **`--gate`** controls how strictly identity fields must match (default 1.0 = exact match on all identity fields).

## Schema

The benchmark uses a 14-section resume schema. Each array section is wrapped as `{sectionTitle, items}` and `personalSummary` as `{sectionTitle, text}`.

| Section | Type | In Headline F1 |
|---------|------|-----------------|
| basics | Singleton (contact info) | Yes |
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

See `src/resume_bench/schema/resume.json` for the full JSON Schema.

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

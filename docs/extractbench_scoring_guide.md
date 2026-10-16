# ExtractBench-Exact Scoring Guide

Binary cell-level scoring for `resume-extract-bench`. Every schema field is a **cell**. Every cell scores **1** (correct) or **0** (incorrect) after normalization. No partial credit.

**Repo:** [`careerflow/resume-extract-bench`](https://github.com/careerflow/resume-extract-bench)
**Dataset:** [`careerflow/ResumeExtractBench`](https://huggingface.co/datasets/careerflow/ResumeExtractBench) (HuggingFace)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Grading Your Own Model](#grading-your-own-model)
5. [How ExtractBench-Exact Scoring Works](#how-extractbench-exact-scoring-works)
6. [Scoring Walkthrough Example](#scoring-walkthrough-example)
7. [Schema Reference](#schema-reference)
8. [Available Pipelines (52)](#available-pipelines-52)
9. [Data Flow](#data-flow)
10. [Key Source Files](#key-source-files)

---

## Quick Start

```bash
# Install
cd resume-extract-bench
pip install -e ".[dev]"

# Download benchmark dataset from HuggingFace
python -m resume_bench.cli download

# Run extraction on two pipelines
python -m resume_bench.cli run llamaextract_agentic_plus openai_gpt-5.6-sol_text

# Grade with ExtractBench-exact (default)
python -m resume_bench.cli grade llamaextract_agentic_plus openai_gpt-5.6-sol_text

# View leaderboard
python -m resume_bench.cli leaderboard
```

---

## Environment Setup

### Installation

```bash
pip install -e ".[dev]"
```

### API Keys

Set as environment variables with `RESUME_BENCH_` prefix, or add to a `.env` file in the repo root (without the prefix):

```bash
export RESUME_BENCH_LLAMA_CLOUD_API_KEY=...
export RESUME_BENCH_OPENAI_API_KEY=...
export RESUME_BENCH_ANTHROPIC_API_KEY=...
export RESUME_BENCH_GOOGLE_API_KEY=...
export RESUME_BENCH_OPENROUTER_API_KEY=...
export RESUME_BENCH_REDUCTO_API_KEY=...
```

### Settings

All settings use the `RESUME_BENCH_` prefix and are loaded via Pydantic (`src/resume_bench/settings.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_BENCH_DATA_DIR` | `~/.cache/resume-bench` | Downloaded dataset location |
| `RESUME_BENCH_OUTPUT_DIR` | `./output` | Extraction results and grades |
| `RESUME_BENCH_CONCURRENCY` | `4` | Max parallel extractions |
| `RESUME_BENCH_RETRIES` | `2` | Retry count for failed extractions |
| `RESUME_BENCH_USE_TRIMMED_SCHEMA` | `true` | Use trimmed section specs (drops fields with 0% GT coverage) |

---

## Step-by-Step Pipeline

### 1. Download Dataset

```bash
python -m resume_bench.cli download
python -m resume_bench.cli download --split test
python -m resume_bench.cli status   # verify what's downloaded
```

Downloads from HuggingFace (`careerflow/ResumeExtractBench`) to `~/.cache/resume-bench/dataset/`:

```
dataset/
  test.jsonl        # each line: {resume_id, files.pdf, ground_truth, difficulty, ...}
  dev.jsonl
  mini.jsonl        # small fixture for testing
  pdfs/
    {resume_id}.pdf
```

Each JSONL line contains embedded ground truth, so no separate GT files are needed.

### 2. Run Extraction

```bash
# Run one or more pipelines
python -m resume_bench.cli run llamaextract_agentic_plus openai_gpt-5.6-sol_text

# Limit to first N resumes (for testing)
python -m resume_bench.cli run openai_gpt-5.6-sol_text --limit 10

# Increase parallelism
python -m resume_bench.cli run openai_gpt-5.6-sol_text --concurrency 8

# Re-extract (ignore cached results)
python -m resume_bench.cli run openai_gpt-5.6-sol_text --no-cache
```

Output per resume at `./output/{pipeline_name}/{split}/{resume_id}.result.json`:

```json
{
  "resume_id": "abc123",
  "pipeline_name": "openai_gpt-5.6-sol_text",
  "output": {
    "basics": {"fname": "Alice", "lname": "Johnson", ...},
    "experience": [{"company": "TechCorp", "position": "Engineer", ...}],
    "skills": [{"name": "Programming", "skills": ["Python", "Java"]}]
  },
  "latency_ms": 3200,
  "cost_usd": 0.0145
}
```

**Input mode routing** (`src/resume_bench/runner/runner.py:59-66`):

| `input_mode` | What happens | Examples |
|-------------|-------------|---------|
| `pdf` | Raw PDF bytes sent to provider | LlamaExtract, Reducto, OpenRouter Gemini |
| `text` | PDF parsed to text first (PyMuPDF or LlamaParse) | OpenAI, Anthropic, Google, OpenRouter text models |
| `images` | PDF pages rasterized to PNG images | OpenRouter Qwen 3.5+, Kimi, GLM, Gemma, DeepSeek |

### 3. Grade

```bash
# ExtractBench-exact scoring (default -- binary 0/1 cells)
python -m resume_bench.cli grade llamaextract_agentic_plus openai_gpt-5.6-sol_text

# Legacy fuzzy scoring (Jaro-Winkler + edit distance, partial credit)
python -m resume_bench.cli grade llamaextract_agentic_plus --fuzzy
```

ExtractBench grades go to `./output/{pipeline_name}/{split}/grades_extractbench/{resume_id}.grade.json`:

```json
{
  "resume_id": "abc123",
  "completed": true,
  "correct": 42,
  "expected": 48,
  "predicted": 45,
  "precision": 0.9333,
  "recall": 0.8750,
  "f1": 0.9032
}
```

Console output per pipeline:

```
openai_gpt-5.6-sol_text
  Micro F1:  0.8531
  Macro F1:  0.8495
  Cells:     12450 / 14200 expected / 13800 predicted
  Completed: 365 / 365
```

### 4. Leaderboard

```bash
python -m resume_bench.cli leaderboard
python -m resume_bench.cli leaderboard --fuzzy  # legacy fuzzy leaderboard
```

```
ExtractBench-Exact Leaderboard (test)
  Rank  Pipeline                        Micro F1  Precision  Recall  Resumes
  1     llamaextract_agentic_plus       0.8812    0.9015     0.8620  365
  2     openai_gpt-5.6-sol_text         0.8531    0.8700     0.8370  365
  ...
```

---

## Grading Your Own Model

### Option A: JSONL Predictions File

Create a JSONL file with one JSON object per line. The `resume_id` must match IDs from the dataset.

```jsonl
{"resume_id": "abc123", "prediction": {"basics": {"fname": "Alice", "lname": "Johnson", "email": "alice@email.com"}, "experience": [{"company": "TechCorp", "position": "Engineer", "startYear": 2021, "description": ["Led team of 5", "Shipped feature X"]}], "skills": [{"skills": ["Python", "Java"]}]}}
{"resume_id": "def456", "prediction": {"basics": {"fname": "Bob", "lname": "Smith"}, "experience": [...]}}
```

The `prediction` field follows the flat schema (see [Schema Reference](#schema-reference)). Alternatively, fields can be top-level (without a `prediction` wrapper):

```jsonl
{"resume_id": "abc123", "basics": {"fname": "Alice", ...}, "experience": [...]}
```

Grade it:

```bash
python -m resume_bench.cli grade-file my_predictions.jsonl
```

Output:

```
ExtractBench-Exact Results (365 resumes)
  Micro F1:        0.8200
  Macro F1:        0.8150
  Micro Precision: 0.8400
  Micro Recall:    0.8010
  Cells Correct:   11500
  Cells Expected:  14350
  Cells Predicted: 13690
  Resumes Graded:  365 / 365
```

### Option B: Python API

```python
from resume_bench.grading.grader import grade_single_extractbench

gt = {
    "basics": {"fname": "Alice", "lname": "Johnson", "email": "alice@email.com"},
    "experience": [
        {
            "company": "TechCorp",
            "position": "Senior Engineer",
            "startYear": 2021,
            "description": ["Led team of 5", "Shipped feature X"],
        }
    ],
}

pred = {
    "basics": {"fname": "Alice", "lname": "Johnson", "email": "alice@email.com"},
    "experience": [
        {
            "company": "TechCorp",
            "position": "Senior Engineer",
            "startYear": 2021,
            "description": ["Led team of 5", "Shipped feature X"],
        }
    ],
}

score = grade_single_extractbench(gt, pred)
print(f"F1:        {score.cells.f1:.4f}")
print(f"Precision: {score.cells.precision:.4f}")
print(f"Recall:    {score.cells.recall:.4f}")
print(f"Correct:   {score.cells.correct} / {score.cells.expected}")
```

### Option C: Low-Level Scoring API

```python
from resume_bench.grading.extractbench import eb_score_resume
from resume_bench.schema.sections import get_sections

cells = eb_score_resume(gt_dict, pred_dict, get_sections())
print(f"Correct: {cells.correct}, Expected: {cells.expected}, Predicted: {cells.predicted}")
print(f"F1: {cells.f1:.4f}")
```

---

## How ExtractBench-Exact Scoring Works

### Core Concept

Every schema field is a **cell**. Each cell is scored binary: **1** (match) or **0** (mismatch) after normalization. No partial credit. The final score is cell-level Precision / Recall / F1.

### String Normalization (`_eb_normalize`)

**Source:** `src/resume_bench/grading/extractbench.py:50-65`

Before comparison, all strings go through `_eb_normalize()`:

1. Strip leading/trailing whitespace
2. Collapse internal whitespace (`\s+` to single space)
3. Lowercase
4. Remove non-semantic punctuation via `[^\w\s+#]` regex (strips quotes, periods, commas, hyphens, parentheses)
5. Keep `+` and `#` (for C++, C#, F#)
6. Collapse spaces again, strip

Examples:

| Raw | Normalized |
|-----|-----------|
| `"Master's of Science"` | `masters of science` |
| `"B.S."` | `bs` |
| `"C++"` | `c++` |
| `"San Francisco, CA"` | `san francisco ca` |
| `"  John   Doe  "` | `john doe` |
| `"+1-555-0101"` | `+1 555 0101` |

### Cell Comparison Rules

| GT Value | Pred Value | Score | Rule |
|----------|-----------|-------|------|
| null/empty | null/empty | **1** | Both absent = correct |
| null/empty | has value | **0** | Hallucination |
| has value | null/empty | **0** | Omission |
| `"TechCorp"` | `"techcorp"` | **1** | Normalized match |
| `"TechCorp"` | `"Tech Corp"` | **0** | Different after normalization |
| `2021` | `2021` | **1** | Exact int match |
| `2021` | `2022` | **0** | Different int |
| `true` | `true` | **1** | Exact bool match |
| `true` | `false` | **0** | Different bool |

**`_is_empty()` definition** (`src/resume_bench/grading/metrics.py:65-75`): `None`, empty strings (after strip), empty lists, or dicts where all values are empty.

### Entity Alignment (Hungarian Algorithm)

**Source:** `src/resume_bench/grading/extractbench.py:148-182`

For sections with multiple entities (experience, education, etc.), GT entities must be matched to predicted entities before cell comparison.

1. **Cost matrix**: For each GT entity `i` and pred entity `j`, count the number of mismatched cells across all scalar fields + description bullets (`_eb_cell_mismatch_count()`).
2. **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`): Find the assignment that minimizes total cell mismatches.
3. **No threshold**: Every assignment from the Hungarian solution is kept, regardless of match quality. This differs from the fuzzy scorer which uses a 0.5 Jaro-Winkler threshold.
4. **Unmatched GT entities**: All their cells count toward `expected` only (hurts recall).
5. **Unmatched pred entities**: All their cells count toward `predicted` only (hurts precision).

### Skills Alignment

**Source:** `src/resume_bench/grading/extractbench.py:265-293`

1. Flatten skill groups into a single list (extract `skills` arrays from all category dicts)
2. Split comma-separated entries (`"Python, Java"` becomes `["Python", "Java"]`) via `_split_comma_skills()`
3. Case-insensitive deduplication
4. Wrap each skill as `{"name": skill_value}` and align via Hungarian on the single `name` field
5. Each matched pair: 1 cell scored by exact match after normalization
6. Missed GT skills: 1 cell added to `expected` per missed skill
7. Hallucinated skills: 1 cell added to `predicted` per spurious skill

### Section Scoring Breakdown

**Source:** `src/resume_bench/grading/extractbench.py:199-398`

The main function `eb_score_resume()` iterates all `SectionSpec` entries and dispatches to three branches:

#### 1. Singleton (`basics`)

- `fname` + `lname` are merged into a single `name` cell (avoids penalizing models that split "Jean Marie" / "Schiraldi" differently from GT)
- Each remaining key field = 1 cell
- Both sides always contribute to `expected` and `predicted` (11 cells for basics: name, prefix, suffix, email, phone, city, state, country, hasPersonalPhoto, linkedinUrl)

#### 2. Flat Skills

See [Skills Alignment](#skills-alignment) above.

#### 3. Entity Lists (experience, education, projects, certifications, awards, volunteering, publications, languages, interests, profiles, customSections)

For each matched entity pair:
- Every scalar field in `SECTION_SCHEMA_FIELDS[section_name]` is 1 cell
- Each description bullet is 1 cell (compared positionally: `bullet[0]` vs `bullet[0]`, etc.)

For description bullets, `expected` and `predicted` are counted independently:
- GT bullet exists at index `i` -> `expected += 1`
- Pred bullet exists at index `i` -> `predicted += 1`
- Both match after normalization -> `correct += 1`

#### 4. personalSummary (special case)

Stored as a plain string (not an entity list). Treated as a single text cell:
- Both empty -> skip entirely (no cell counted)
- Only GT has text -> `expected += 1` (omission penalty)
- Only pred has text -> `predicted += 1` (hallucination penalty)
- Both have text -> 1 cell, scored by normalized exact match

### Aggregation Formulas

```
Per resume:
  precision = correct / predicted        (0.0 if predicted = 0)
  recall    = correct / expected         (0.0 if expected = 0)
  f1        = 2 * P * R / (P + R)       (0.0 if P + R = 0)

Per model (across all resumes):
  micro_f1   = F1(sum_correct, sum_expected, sum_predicted)
  macro_f1   = mean(per_resume_f1)       <-- primary reported metric
```

---

## Scoring Walkthrough Example

### Input

**GT:**
```json
{
  "basics": {"fname": "Alice", "lname": "Johnson", "email": "alice@email.com", "phone": "+1-555-0101"},
  "experience": [
    {"company": "TechCorp", "position": "Engineer", "startYear": 2021, "description": ["Led team", "Shipped v2"]}
  ],
  "skills": [{"skills": ["Python", "Java"]}]
}
```

**Prediction:**
```json
{
  "basics": {"fname": "Alice", "lname": "Johnson", "email": "alice@email.com", "phone": "+1 555 0101"},
  "experience": [
    {"company": "TechCorp", "position": "Engineer", "startYear": 2021, "description": ["Led team"]}
  ],
  "skills": [{"skills": ["Python", "JavaScript"]}]
}
```

### Cell-by-Cell Scoring

**basics** (each field = 1 cell):

| Cell | GT | Pred | Normalized GT | Normalized Pred | Score |
|------|-----|------|--------------|-----------------|-------|
| name | Alice Johnson | Alice Johnson | `alice johnson` | `alice johnson` | **1** |
| email | alice@email.com | alice@email.com | `alice@emailcom` | `alice@emailcom` | **1** |
| phone | +1-555-0101 | +1 555 0101 | `+1 555 0101` | `+1 555 0101` | **1** |
| prefix | null | null | | | **1** |
| suffix | null | null | | | **1** |
| city | null | null | | | **1** |
| state | null | null | | | **1** |
| country | null | null | | | **1** |
| hasPersonalPhoto | null | null | | | **1** |
| linkedinUrl | null | null | | | **1** |

Basics subtotal: **10 correct / 10 expected / 10 predicted**

**experience** (aligned: TechCorp <-> TechCorp):

Scalar fields (13 total, most are null/null):

| Cell | GT | Pred | Score |
|------|-----|------|-------|
| company | TechCorp | TechCorp | **1** |
| position | Engineer | Engineer | **1** |
| startYear | 2021 | 2021 | **1** |
| startMonth...companyUrl | null | null | **1** each (10 null/null fields) |

Description bullets (positional):

| Index | GT | Pred | Score | expected | predicted |
|-------|-----|------|-------|----------|-----------|
| 0 | "Led team" | "Led team" | **1** | +1 | +1 |
| 1 | "Shipped v2" | (missing) | **0** | +1 | +0 |

Experience subtotal: **14 correct / 15 expected / 14 predicted**

**skills** (aligned via Hungarian):

| GT Skill | Pred Skill | Match? |
|----------|-----------|--------|
| Python | Python | **1** |
| Java | JavaScript | **0** (normalized: `java` vs `javascript`) |

Skills subtotal: **1 correct / 2 expected / 2 predicted**

### Final Score

```
correct  = 10 + 14 + 1 = 25
expected = 10 + 15 + 2 = 27
predicted = 10 + 14 + 2 = 26

precision = 25 / 26 = 0.9615
recall    = 25 / 27 = 0.9259
f1        = 2 * 0.9615 * 0.9259 / (0.9615 + 0.9259) = 0.9434
```

---

## Schema Reference

### 14 Scored Sections

| Section | Kind | Key Fields (alignment) | Description Scored? |
|---------|------|----------------------|---------------------|
| `basics` | singleton | prefix, fname, lname, suffix, email, phone, city, state, country, hasPersonalPhoto, linkedinUrl | n/a |
| `experience` | entity-list | company, position | Yes |
| `education` | entity-list | institution, area | Yes |
| `projects` | entity-list | name | Yes |
| `personalSummary` | text cell | text | n/a |
| `certifications` | entity-list | name, issuer | No |
| `awards` | entity-list | title | No |
| `volunteering` | entity-list | organization, position | Yes |
| `skills` | flat-list | (flat skill names) | n/a |
| `publications` | entity-list | name | No |
| `languages` | entity-list | name | No |
| `interests` | entity-list | name | No |
| `profiles` | entity-list | network | No |
| `customSections` | entity-list | sectionTitle | Yes |

### Scalar Fields Per Section (`SECTION_SCHEMA_FIELDS`)

These are the fields scored as individual cells per entity in ExtractBench-exact mode. Defined in `src/resume_bench/grading/metrics.py:16-44`.

```
experience (13):    company, position, startMonth, startYear, endMonth, endYear,
                    currentlyWorkHere, city, state, country, isRemote,
                    roleDescription, companyUrl

education (12):     institution, area, studyType, score, startMonth, startYear,
                    endMonth, endYear, currentlyStudyHere, city, state, country

projects (11):      name, startYear, endYear, url, companyName, city, state,
                    country, startMonth, endMonth, inProgress

certifications (4): name, issuer, date, url

awards (5):         title, awarder, date, summary, url

volunteering (11):  organization, position, startYear, endYear,
                    currentlyVolunteerHere, startMonth, endMonth,
                    city, state, country, summary

publications (4):   name, publisher, date, summary

languages (1):      name

interests (1):      name

profiles (2):       network, url

customSections (2): sectionTitle, summary

personalSummary (1): text
```

Plus description bullets (variable count per entity) for experience, education, projects, volunteering, and customSections.

### Example Extraction Output

```json
{
  "basics": {
    "prefix": null,
    "fname": "Alice",
    "lname": "Johnson",
    "suffix": null,
    "email": "alice.johnson@email.com",
    "phone": "+1-555-0101",
    "city": "San Francisco",
    "state": "California",
    "country": "United States",
    "hasPersonalPhoto": false,
    "linkedinUrl": "https://linkedin.com/in/alicejohnson"
  },
  "experience": [
    {
      "company": "TechCorp",
      "position": "Senior Software Engineer",
      "startMonth": 3,
      "startYear": 2021,
      "endMonth": null,
      "endYear": null,
      "currentlyWorkHere": true,
      "city": "San Francisco",
      "state": "CA",
      "country": "USA",
      "isRemote": false,
      "roleDescription": null,
      "companyUrl": null,
      "description": [
        "Led development of microservices architecture",
        "Managed team of 5 engineers across 2 time zones",
        "Reduced API response time by 40%"
      ]
    }
  ],
  "education": [
    {
      "institution": "Stanford University",
      "area": "Computer Science",
      "studyType": "Bachelor of Science",
      "score": "3.9",
      "startMonth": 9,
      "startYear": 2013,
      "endMonth": 6,
      "endYear": 2017,
      "currentlyStudyHere": false,
      "city": "Stanford",
      "state": "CA",
      "country": "USA"
    }
  ],
  "skills": [
    {"name": "Programming Languages", "skills": ["Python", "Java", "TypeScript"]},
    {"name": "Tools", "skills": ["Docker", "Kubernetes", "AWS"]}
  ],
  "projects": [],
  "certifications": [],
  "awards": [],
  "volunteering": [],
  "publications": [],
  "languages": [{"name": "English"}, {"name": "Spanish"}],
  "interests": [],
  "profiles": [{"network": "GitHub", "url": "https://github.com/alicejohnson"}],
  "customSections": [],
  "personalSummary": "Experienced software engineer with 5+ years building scalable systems."
}
```

---

## Available Pipelines (52)

List all with: `python -m resume_bench.cli providers`

### LlamaExtract (PDF-native)

| Pipeline | Tier |
|----------|------|
| `llamaextract_agentic_plus` | Highest quality |
| `llamaextract_agentic` | Balanced |
| `llamaextract_cost_effective` | Fastest |
| `llamaextract_turbo` | Ultra-fast |

### Reducto (PDF-native)

| Pipeline |
|----------|
| `reducto_extract` |

### OpenAI (text mode)

| Pipeline | Model |
|----------|-------|
| `openai_gpt-6-astra_text` | GPT-6 Astra |
| `openai_gpt-5.6-sol_text` | GPT-5.6 Sol |
| `openai_gpt-5.6-terra_text` | GPT-5.6 Terra |
| `openai_gpt-5.6-luna_text` | GPT-5.6 Luna |
| `openai_gpt-5.5_text` | GPT-5.5 |
| `openai_gpt-5.4_text` | GPT-5.4 |
| `openai_gpt-5.4-nano_text` | GPT-5.4 Nano |
| `openai_gpt-5.4-mini_text` | GPT-5.4 Mini |
| `openai_gpt-4.1_text` | GPT-4.1 |
| `openai_gpt-4.1-mini_text` | GPT-4.1 Mini |
| `openai_gpt-4.1-nano_text` | GPT-4.1 Nano |
| `openai_gpt-4o_text` | GPT-4o |

### Anthropic (text mode)

| Pipeline | Model |
|----------|-------|
| `anthropic_claude-opus-4-8_text` | Claude Opus 4.8 |
| `anthropic_claude-opus-5_text` | Claude Opus 5 |
| `anthropic_claude-sonnet-4-6_text` | Claude Sonnet 4.6 |
| `anthropic_claude-sonnet-5_text` | Claude Sonnet 5 |
| `anthropic_claude-haiku-4-5_text` | Claude Haiku 4.5 |

### Google (text mode)

| Pipeline | Model |
|----------|-------|
| `google_gemini-3.8-flash_text` | Gemini 3.8 Flash |
| `google_gemini-3.6-flash_text` | Gemini 3.6 Flash |
| `google_gemini-3.5-flash_text` | Gemini 3.5 Flash |
| `google_gemini-3.1-pro_text` | Gemini 3.1 Pro |
| `google_gemini-3.1-flash-lite_text` | Gemini 3.1 Flash Lite |
| `google_gemini-2.5-pro_text` | Gemini 2.5 Pro |
| `google_gemini-2.5-flash_text` | Gemini 2.5 Flash |

### OpenRouter -- Vision/Images

| Pipeline | Model | Input |
|----------|-------|-------|
| `openrouter_qwen38_flash` | Qwen 3.8 Flash | Images |
| `openrouter_qwen38_flash_next` | Qwen 3.8 Flash Next | Images |
| `openrouter_qwen38_27b` | Qwen 3.8 27B | Images |
| `openrouter_qwen36_35b` | Qwen 3.6 35B | Images |
| `openrouter_qwen35_35b` | Qwen 3.5 35B | Images |
| `openrouter_qwen35_9b` | Qwen 3.5 9B | Images |
| `openrouter_kimi_k3` | Kimi K3 | Images |
| `openrouter_kimi_k2` | Kimi K2 | Images |
| `openrouter_kimi_vl_a3b` | Kimi VL A3B | Images |
| `openrouter_glm53_flash` | GLM 5.3 Flash | Images |
| `openrouter_gemma4_26b` | Gemma 4 26B | Images |
| `openrouter_gemma4_e4b` | Gemma 4 E4B | Images |
| `openrouter_gemma3_27b` | Gemma 3 27B | Images |
| `openrouter_internvl35_14b` | InternVL3 14B | Images |
| `openrouter_minicpm_v45` | MiniCPM V4.5 | Images |
| `openrouter_deepseek_v41_flash` | DeepSeek V4.1 Flash | Images |

### OpenRouter -- Text (LlamaParse pre-processing)

| Pipeline | Model |
|----------|-------|
| `openrouter_qwen3_235b` | Qwen 3.0 235B |
| `openrouter_qwen3_30b` | Qwen 3.0 30B |
| `openrouter_qwen25_72b` | Qwen 2.5 72B |
| `openrouter_qwen25_7b` | Qwen 2.5 7B |
| `openrouter_glm4_32b` | GLM-4 32B |

### OpenRouter -- PDF-native

| Pipeline | Model |
|----------|-------|
| `openrouter_gemini38_flash` | Gemini 3.8 Flash |
| `openrouter_gemini35_flash` | Gemini 3.5 Flash |

---

## Data Flow

```
HuggingFace (careerflow/ResumeExtractBench)
    |
    v  download
~/.cache/resume-bench/dataset/
    +-- test.jsonl           # {resume_id, files.pdf, ground_truth, difficulty, ...}
    +-- dev.jsonl
    +-- mini.jsonl
    +-- pdfs/
        +-- {resume_id}.pdf
    |
    v  run
./output/{pipeline_name}/{split}/
    +-- {resume_id}.result.json    # {resume_id, output, latency_ms, cost_usd, ...}
    |
    v  grade
./output/{pipeline_name}/{split}/grades_extractbench/
    +-- {resume_id}.grade.json     # {resume_id, correct, expected, predicted, f1, ...}
    |
    v  leaderboard
Console output: ranked table of pipeline F1 scores
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/resume_bench/cli.py` | CLI entry point: `download`, `run`, `grade`, `leaderboard`, `grade-file`, `validate` |
| `src/resume_bench/grading/extractbench.py` | ExtractBench scorer: `eb_score_resume()`, `_eb_normalize()`, `_eb_align_entities()`, `_split_comma_skills()` |
| `src/resume_bench/grading/grader.py` | Orchestrator: `grade_single_extractbench()`, `grade_pipelines_extractbench()`, `grade_single()` (fuzzy) |
| `src/resume_bench/grading/models.py` | Data classes: `CellCounts`, `ExtractBenchScore`, `ResumeScore`, `SectionScore` |
| `src/resume_bench/grading/metrics.py` | `SECTION_SCHEMA_FIELDS`, `_is_empty()`, `_split_comma_skills()`, `_positional_bullet_score()` |
| `src/resume_bench/grading/text.py` | `edit_distance_ratio()`, `field_similarity()` (Jaro-Winkler), `token_f1()` |
| `src/resume_bench/grading/alignment.py` | `align_entities()` -- Hungarian alignment for fuzzy scoring |
| `src/resume_bench/schema/sections.py` | `SECTIONS_FULL`, `SECTIONS_TRIMMED`, `SectionSpec`, `get_sections()` |
| `src/resume_bench/schema/resume_v1.json` | Full JSON schema definition |
| `src/resume_bench/runner/runner.py` | Extraction runner: `run_pipelines()`, caching, concurrency |
| `src/resume_bench/providers/pipelines.py` | All 52 pipeline definitions (`PipelineSpec` list) |
| `src/resume_bench/providers/base.py` | `PipelineSpec`, `ExtractionRequest`, `RunRecord`, `InputMode` |
| `src/resume_bench/prompts/extraction_v1.py` | System prompt sent to LLM-based extraction models |
| `src/resume_bench/dataset/loader.py` | HuggingFace download + JSONL loading: `download_dataset()`, `load_split()` |
| `src/resume_bench/dataset/models.py` | `TestCase` dataclass (resume_id, pdf_path, ground_truth, difficulty) |
| `src/resume_bench/settings.py` | Pydantic settings with `RESUME_BENCH_` env prefix |

### ExtractBench vs Fuzzy Scoring -- Which to Use

| | ExtractBench-exact (default) | Fuzzy (`--fuzzy`) |
|---|---|---|
| **Cell scoring** | Binary 0/1 after normalization | Partial credit (Jaro-Winkler 0.0-1.0) |
| **Description bullets** | Positional exact match per bullet | Positional edit distance ratio |
| **Entity alignment** | Hungarian on cell-mismatch count, no threshold | Hungarian on Jaro-Winkler similarity, 0.5 threshold |
| **Skills** | Flatten + comma-split + dedup + Hungarian exact | Flatten + comma-split + dedup + Hungarian fuzzy |
| **personalSummary** | Single cell, exact match | Edit distance ratio |
| **Primary metric** | `macro_f1` (mean of per-resume F1) | `macro_entity_f1` (mean of per-section F1 means) |
| **Use when** | Benchmarking, leaderboards, publication | Debugging, understanding partial matches |

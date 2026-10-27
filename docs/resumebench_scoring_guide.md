# ResumeBench-Strict Scoring Guide

Record-level binary scoring for `resume-extract-bench`. An entity is ONE unit: it is correct only if every field matches. No partial credit.

**Repo:** [`careerflow/resume-extract-bench`](https://github.com/careerflow/resume-extract-bench)
**Dataset:** [`ResumeBench`](https://huggingface.co/datasets/careerflow/ResumeExtractBench) (HuggingFace)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Grading Your Own Model](#grading-your-own-model)
5. [How ResumeBench-Strict Scoring Works](#how-resumebench-strict-scoring-works)
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

# Grade with ResumeBench-Strict (default)
python -m resume_bench.cli grade llamaextract_agentic_plus openai_gpt-5.6-sol_text

# Grade with verbatim mode (case/punctuation sensitive)
python -m resume_bench.cli grade llamaextract_agentic_plus --verbatim

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

---

## Step-by-Step Pipeline

### 1. Download Dataset

```bash
python -m resume_bench.cli download
python -m resume_bench.cli status
```

### 2. Run Extraction

```bash
python -m resume_bench.cli run llamaextract_agentic_plus --limit 10
```

### 3. Grade

```bash
python -m resume_bench.cli grade llamaextract_agentic_plus
python -m resume_bench.cli grade llamaextract_agentic_plus --verbatim
```

### 4. Leaderboard

```bash
python -m resume_bench.cli leaderboard
```

---

## Grading Your Own Model

### Option A: JSONL Predictions File

```jsonl
{"resume_id": "abc123", "prediction": {"basics": {"fname": "Alice", ...}, "experience": [...]}}
```

```bash
python -m resume_bench.cli grade-file my_predictions.jsonl --verbatim
```

### Option B: Python API

```python
from resume_bench.grading.grader import grade_single_strict

result = grade_single_strict(gt, pred, verbatim=True)
print(f"F1: {result['f1']:.4f}")
```

### Option C: Low-Level Scoring API

```python
from resume_bench.grading.strict import score_resume_strict
from resume_bench.schema.sections import get_sections

result = score_resume_strict(gt, pred, get_sections(), verbatim=True)
```

---

## How ResumeBench-Strict Scoring Works

### Core Rules

1. **Record-level units.** An entity is ONE unit. Correct only if aligned AND every non-empty field matches AND bullets match one-for-one.

2. **Empty is not free.** Both sides null/empty does NOT count as correct.

3. **Section-averaged.** F1 is computed per section and averaged, so every section weighs the same.

4. **Verbatim text** (with `--verbatim`). Case and punctuation count. Only typographic variants are folded.

### Standard vs Verbatim Normalization

**Standard mode** (`_rb_normalize`): lowercase, strip punctuation (keep `+`, `#`), collapse whitespace.

**Verbatim mode** (`verbatim_normalize`): fold typographic variants only (smart quotes, em dashes), collapse whitespace. Case and punctuation preserved.

### Entity Alignment (Hungarian Algorithm)

GT entities matched to predicted entities via cell-mismatch cost matrix + Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). No threshold — every assignment kept.

### Record-Level Scoring

Each matched pair: all fields must match AND bullet count must match AND each bullet[i] must match. One wrong field = entire entity scores 0.

---

## Scoring Walkthrough Example

**GT:**
```json
{
  "experience": [
    {"company": "TechCorp", "position": "Engineer", "description": ["Led team", "Shipped v2"]},
    {"company": "StartupXYZ", "position": "Intern", "description": ["Built API"]}
  ]
}
```

**Prediction:**
```json
{
  "experience": [
    {"company": "TechCorp", "position": "Engineer", "description": ["Led team"]}
  ]
}
```

- TechCorp pair: all scalars match, but GT has 2 bullets, pred has 1 → **FAIL** (entire entity = 0)
- StartupXYZ: missed → expected only

Experience: correct=0, expected=2, predicted=1 → F1=0.0

---

## Schema Reference

### 14 Scored Sections

| Section | Kind | Key Fields |
|---------|------|------------|
| `basics` | singleton | prefix, fname, lname, suffix, email, phone, city, state, country, hasPersonalPhoto, linkedinUrl |
| `experience` | entity-list | company, position + 11 fields + description |
| `education` | entity-list | institution, area + 10 fields + description |
| `projects` | entity-list | name + 10 fields + description |
| `personalSummary` | text cell | text |
| `certifications` | entity-list | name, issuer, date, url |
| `awards` | entity-list | title, awarder, date, summary, url |
| `volunteering` | entity-list | organization, position + 9 fields + description |
| `skills` | flat-list | flat skill names |
| `publications` | entity-list | name, publisher, date, summary |
| `languages` | entity-list | name |
| `interests` | entity-list | name |
| `profiles` | entity-list | network, url |
| `customSections` | entity-list | sectionTitle, summary + description |

---

## Data Flow

```
HuggingFace dataset → download → run (extraction) → grade (strict) → leaderboard
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/resume_bench/grading/strict.py` | `score_resume_strict()`, `verbatim_normalize()`, `SECTION_SCHEMA_FIELDS` |
| `src/resume_bench/grading/grader.py` | `grade_single_strict()`, `grade_pipelines_strict()` |
| `src/resume_bench/cli.py` | CLI: download, run, grade, leaderboard, grade-file, validate |
| `src/resume_bench/schema/sections.py` | Section specs and schema definitions |

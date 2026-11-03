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

# Grade (verbatim is the default; use --no-verbatim for case-insensitive)
python -m resume_bench.cli grade llamaextract_agentic_plus

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
python -m resume_bench.cli grade llamaextract_agentic_plus --no-verbatim  # case-insensitive
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
python -m resume_bench.cli grade-file my_predictions.jsonl  # verbatim is the default
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
from resume_bench.schema.sections import SECTIONS

result = score_resume_strict(gt, pred, SECTIONS, verbatim=True)
```

---

## How ResumeBench-Strict Scoring Works

### Core Rules

1. **Per-field units.** Each field in a matched entity is scored independently — partial credit for records that match some fields but not all.

2. **Identity matching.** Entities are matched by identity fields (e.g. company + position + years) using greedy matching. The `--gate` threshold (default 1.0) controls how strictly identity fields must match.

3. **Empty is not free.** Both sides null/empty does NOT count as correct.

4. **Section-averaged.** F1 is computed per section and averaged, so every section weighs the same.

5. **Verbatim text (default).** Case and punctuation count. Only typographic variants are folded. Contacts and locations compared case-insensitively. Use `--no-verbatim` for standard mode.

6. **Failed extractions score 0.** Crashes are penalized, not skipped.

### Standard vs Verbatim Normalization

**Standard mode** (`_rb_normalize`): lowercase, strip punctuation (keep `+`, `#`), collapse whitespace.

**Verbatim mode** (`verbatim_normalize`): fold typographic variants only (smart quotes, em dashes), collapse whitespace. Case and punctuation preserved.

### Entity Alignment

Structured sections (experience, education, projects, etc.) use **identity-field greedy matching** — entities are matched by their identity fields (company + position + years, etc.) with a configurable gate. Keyword sections (languages, interests, profiles, publications) use **Hungarian algorithm** alignment.

### Per-Field Scoring

Each matched pair is scored per-field. Scalar fields: 1 if equal, 0 otherwise. Description bullets compared positionally — bullet[0] vs bullet[0], each scoring 1 or 0. Missing/extra bullets score 0.

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

**TechCorp (matched):** company=1, position=1, bullet[0]=1, bullet[1]=0 (missing) → correct=3, expected=4, predicted=3

**StartupXYZ (missed):** all fields are expected-only → correct=0, expected=3, predicted=0

**Experience totals:** correct=3, expected=7, predicted=3 → Precision=1.0, Recall=3/7=0.43, F1=0.60

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

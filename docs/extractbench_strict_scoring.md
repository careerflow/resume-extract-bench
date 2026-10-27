# ExtractBench-Strict

A harder companion to ExtractBench-exact (`resume_bench.grading.extractbench`).
Implementation: `resume_bench.grading.strict` (`score_resume_strict`).

## Why

On large benchmark sets, the top parsers score 0.95-0.97 with ExtractBench-exact,
compressing the leaderboard at the top. Small but real differences (a missing job,
a dropped bullet, a lost certification list) barely move the number. Three
properties of the cell metric cause most of that:

- **Partial entries earn most of their cells.** A job with a wrong end date and a
  missing bullet still scores ~90% of its cells.
- **Empty fields are free.** A field empty in both the label and the output
  (prefix, suffix, URLs, `isRemote`, scores, ...) is a correct cell for every
  parser -- points that measure nothing.
- **Long sections dominate.** Experience bullets are most of a resume's cells, so
  missing a whole certifications or custom section moves the document F1 by only
  a few points.

## Rules

Same inputs, `normalize_flat`, `_eb_normalize` string normalization, Hungarian
entity alignment (`_eb_align_entities`), skills handling (comma split, dedupe)
and per-document averaging as ExtractBench-exact. Changed:

1. **Record-level units.** Each entity (job, school, project, certification,
   award, volunteer role, publication, language, interest, profile, custom
   section, summary) is one unit. It is correct only if it is aligned **and**
   every field that is non-empty on either side matches **and** its bullets
   match one-for-one (same count, each equal). Expected = label entities,
   predicted = parser entities.
2. **Empty is not free.** Basics fields (name = fname + lname, prefix, suffix,
   email, phone, city, state, country, photo, LinkedIn) count only where the
   label or the parser has a value. Skills: one unit per skill (as in
   ExtractBench-exact).
3. **Section-averaged.** Per document, precision / recall / F1 are computed per
   section (sections present in the label or the output) and averaged, so every
   section weighs the same. Documents are then averaged (macro over resumes),
   as before.

## Benchmark v3: verbatim text (`verbatim=True`)

4. **Verbatim text.** Strings match only if equal after whitespace collapsing --
   case and punctuation count. ExtractBench-exact lowercases and strips
   punctuation (`Master's` = `Masters`, `B.S.` = `BS`); the extraction prompt
   asks for VERBATIM copies, so v3 holds parsers to it. Typographic variants of
   one character are not errors: all hyphens / dashes (- ‐ ‑ ‒ – — ― −) count
   as "-", all apostrophes / single quotes as "'", all double quotes as '"',
   soft hyphens are dropped.

## Usage

```python
from resume_bench.grading.grader import grade_single_strict

# ExtractBench-Strict (v2) -- case-insensitive, same normalization as exact
result = grade_single_strict(ground_truth, prediction)

# ExtractBench-Strict Verbatim (v3) -- case and punctuation sensitive
result = grade_single_strict(ground_truth, prediction, verbatim=True)

# Result shape:
# {
#   "f1": 0.85,
#   "precision": 0.88,
#   "recall": 0.82,
#   "correct": 42,
#   "expected": 50,
#   "predicted": 48,
#   "sections": {
#     "basics": {"f1": 0.9, "precision": 0.9, "recall": 0.9, "correct": 9, "expected": 10, "predicted": 10},
#     "experience": {"f1": 0.8, ...},
#     ...
#   }
# }
```

Or use `score_resume_strict` directly for custom section specs:

```python
from resume_bench.grading.strict import score_resume_strict
from resume_bench.schema.sections import get_sections

result = score_resume_strict(gt, pred, get_sections(), verbatim=True)
```

## Properties

- Identical output and label -> 1.0. Any error in an entry makes that whole
  entry wrong.
- Same alignment and normalization as ExtractBench-exact, so the two metrics
  disagree only through the three rules above.
- Output includes per-section `{correct, expected, predicted, f1, precision,
  recall}` for section-level analysis.

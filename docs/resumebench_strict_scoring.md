# ResumeBench-Strict

The scoring method for `resume-extract-bench`.
Implementation: `resume_bench.grading.strict` (`score_resume_strict`).

## Rules

Three rules define the scoring:

1. **Record-level units.** Each entity (job, school, project, certification,
   award, volunteer role, publication, language, interest, profile, custom
   section, summary) is one unit. It is correct only if it is aligned **and**
   every field that is non-empty on either side matches **and** its bullets
   match one-for-one (same count, each equal). Expected = label entities,
   predicted = parser entities.
2. **Empty is not free.** Basics fields (name = fname + lname, prefix, suffix,
   email, phone, city, state, country, photo, LinkedIn) count only where the
   label or the parser has a value. Skills: one unit per skill.
3. **Section-averaged.** Per document, precision / recall / F1 are computed per
   section (sections present in the label or the output) and averaged, so every
   section weighs the same. Documents are then averaged (macro over resumes).

## Verbatim mode (`verbatim=True`)

An additional fourth rule:

4. **Verbatim text.** Strings match only if equal after whitespace collapsing --
   case and punctuation count. Standard mode lowercases and strips punctuation
   (`Master's` = `Masters`, `B.S.` = `BS`); verbatim mode holds parsers to
   exact reproduction. Typographic variants of one character are not errors: all
   hyphens / dashes (- ‐ ‑ ‒ – — ― −) count as "-", all apostrophes / single
   quotes as "'", all double quotes as '"', soft hyphens are dropped.

## Usage

```python
from resume_bench.grading.grader import grade_single_strict

# Standard mode -- case-insensitive, punctuation stripped
result = grade_single_strict(ground_truth, prediction)

# Verbatim mode -- case and punctuation sensitive
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
- Uses Hungarian algorithm alignment (`_rb_align_entities`) for entity matching,
  comma-split skill deduplication, and `_rb_normalize` string normalization.
- Output includes per-section `{correct, expected, predicted, f1, precision,
  recall}` for section-level analysis.

# ResumeBench-Strict

The scoring method for `resume-extract-bench`.
Implementation: `resume_bench.grading.strict` (`score_resume_strict`).

## Rules

Six rules define the scoring:

1. **Per-field units.** Each field in a matched entity is scored independently —
   partial credit for records that match some fields but not all. Description
   bullets are compared positionally (bullet[0] vs bullet[0], etc.); missing
   or extra bullets score 0.
2. **Identity matching.** Entities in structured sections (experience, education,
   projects, etc.) are matched by identity fields (company + position + years,
   institution + studyType + years, etc.) using direct greedy matching with a
   configurable gate (default 1.0 = all identity fields must match). Keyword
   sections (languages, interests, profiles, publications) use Hungarian alignment.
3. **Empty is not free.** Basics fields (name = fname + lname, prefix, suffix,
   email, phone, city, state, country, photo, LinkedIn) count only where the
   label or the parser has a value. Skills: one unit per skill.
4. **Section-averaged.** Per document, precision / recall / F1 are computed per
   section (sections present in the label or the output) and averaged, so every
   section weighs the same. Documents are then averaged (macro over resumes).
5. **Failed extractions score 0.** A provider that crashes on a resume gets F1=0
   for that resume (not skipped), penalizing unreliable providers.
6. **Name composition.** The ``basics`` name is composed from
   ``prefix + fname + lname + suffix``, whitespace-collapsed.

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
from resume_bench.schema.sections import SECTIONS

result = score_resume_strict(gt, pred, SECTIONS, verbatim=True)
```

## Properties

- Identical output and label -> 1.0. Any error in an entry makes that whole
  entry wrong.
- Uses Hungarian algorithm alignment (`_rb_align_entities`) for entity matching,
  comma-split skill deduplication, and `_rb_normalize` string normalization.
- Output includes per-section `{correct, expected, predicted, f1, precision,
  recall}` for section-level analysis.

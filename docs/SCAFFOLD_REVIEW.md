# Review — `resume-extract-bench` scaffold (commit `55bc7e0`)

**Reviewed:** 2026-08-31 · 42 tracked files, ~2,550 lines · against `docs/RESTRUCTURING_PLAN.md` v1.1 and `docs/ACTION_PLAN.md` tickets B-01…B-13.
**Method:** read every source file; diffed the vendored schema and prompt against the monolith; executed the grader with a stubbed similarity function (no PyPI access here, and the repo's `.venv` is a macOS Python 3.14 venv, so the tests could not be run in this environment).

## Verdict

The skeleton is the right shape — package layout, `PipelineSpec` / `Provider` / registry, `SectionSpec`, Typer CLI, settings, vacuous-section handling and the error hierarchy all match the plan. But it is **not runnable end-to-end yet and, as written, would grade every real provider at ≈0**. Five blockers below must be fixed before any number from this repo is trusted. Estimated fix effort: 3–4 person-days, then re-review.

---

## A. Blockers (wrong results or crashes)

### A1. Grader never converts provider output → grades ≈ 0 for every real pipeline
`runner/_run_single` stores `raw_output = provider.extract(req)`. Every provider returns `{"parsed": {...}, "model": ..., "usage": ...}`. `grading/grader._load_pipeline_results` then feeds that **wrapper** straight into `grade_single`, which looks up `prediction["experience"]` etc. and finds nothing. `Provider.to_canonical()` is never called anywhere.
Verified with the bundled fixture: `grade_single(gt, pred)` → macro F1 **1.0**; `grade_single(gt, {"parsed": pred})` (what the runner actually writes) → **0.02**.
**Fix:** add `output: dict | None` to `RunRecord` (the plan had it); runner sets `output = provider.to_canonical(raw)`; grader reads `record["output"]`, not `raw_output`. Keep `raw_output` for re-normalization.

### A2. Failed / missing runs are skipped instead of scored 0
`grade_pipelines`: `if pred is None: errors += 1; continue` and `_load_pipeline_results` drops records with `error` or empty `raw_output`. Headline F1 is therefore computed on successes only — the opposite of the agreed policy (ExtractBench: "failed and missing documents score zero"). `ResumeScore.completed` exists but is never set to `False`.
**Fix:** for every GT resume without a valid output, append `ResumeScore(completed=False)` with all sections at 0 (non-vacuous); report both `resume_entity_f1` (penalized) and `completed_only_f1`.

### A3. Schema was *not* vendored verbatim — ~40 field descriptions were shortened
`schema/resume_v1.json` has identical structure to `RESUME_EXTRACTION_SCHEMA` but every `description` string was rewritten shorter, e.g.
`fname`: "First name of the person. Split the full name into first and last — everything …" → "First name of the person".
Those descriptions are extraction *instructions* consumed by LlamaExtract and by every LLM prompt; changing them changes results and breaks comparability with the monolith's runs.
**Fix:** regenerate the file mechanically: `json.dump(RESUME_EXTRACTION_SCHEMA)` + add only `$schema`, `$id`, `x-benchmark-version`. Add a test that asserts equality with a checked-in hash.

### A4. Prompt was also trimmed (4,607 → 3,797 chars)
`prompts/extraction_v1.txt` drops several rules from `EXTRACTION_SYSTEM_PROMPT`: "professional memberships → certifications", "extract from education sections if they are certifications", "do not stop early even if the skills list is very long", "include methodologies/soft skills/languages", "only portrait/headshot photos — not logos, icons, badges", "do not extract board roles as awards". Em-dashes/arrows were also replaced.
**Fix:** copy verbatim; same hash test.

### A5. LlamaExtract and Reducto providers call APIs that don't match the working monolith code
- `llamaextract.py`: `client.extraction.create_extraction(schema=…, input_files=[…], config={"tier": tier})`. The monolith (which works) does `client.files.create(file=f, purpose="extract")` then `client.extract.run(file_input=file_id, configuration={"data_schema", "tier", "confidence_scores": True, "system_prompt"})`. The new code also drops `system_prompt` and `confidence_scores`, so LlamaExtract would run **without** the shared prompt — unfair vs LLM pipelines.
- `reducto.py`: `client.extract(document_url=str(local_path), schema=…)` — a local path is not a URL. Monolith: `client.upload(file=f)` → `client.extract.run(input={"type": "file_id", …}, instructions={"schema": …})` plus the result-unwrapping logic (SDK returns list/model/dict depending on version).
**Fix:** port the monolith's `extract()` bodies (they are the tested ones); keep the new error mapping.

---

## B. Plan deviations that must be reconciled (not crashes, but wrong methodology)

| # | Issue | Where | Fix |
|---|---|---|---|
| B1 | **Basics is inside the headline macro.** `score_singleton` returns `f1 = mean field accuracy` and `grade_single` puts it in `sections`, so `macro_entity_f1` averages 9 sections. Plan: headline = 7 entity sections + skills; basics reported separately as *Basics Field Accuracy* | `grading/metrics.py`, `grading/models.py` | Exclude `SectionKind.SINGLETON` from `macro_entity_f1`; expose `basics_field_accuracy` on `ResumeScore` |
| B2 | No canonicalization layer (`normalize_flat` equivalent). `to_canonical` returns raw parsed JSON; monolith's `_clean_val`, description unwrapping, and `descriptions`→`description` handling are gone. Verified: a prediction using `descriptions` scores description F1 = 0 | plan B-03 | Add `grading/canonical.py::to_canonical(dict) -> dict` (port `normalize_flat`) and call it in every provider's `to_canonical` |
| B3 | `personalSummary` special-cased in `grade_single` (token-F1 as P=R=F1) while `SECTIONS` declares it `ENTITY_LIST` with key `("text",)` — spec and code disagree; monolith used JW on a 1-item list | `grader.py:60-85` | Pick one, encode it in `SectionKind` (e.g. `TEXT_BLOCK`) and document in methodology |
| B4 | Micro Entity F1 (pooled counts) not computed; no bootstrap CI; no threshold sensitivity | plan §C.5 item 0, §C.8 | `report/aggregate.py` |
| B5 | Cache key ignores schema/prompt. `runner/cache.py::cache_key` exists but is unused; results are keyed only by `resume_id`, so editing the prompt silently reuses stale outputs | `runner.py` | Write `_metadata.json` per run dir with schema+prompt hashes; invalidate when they differ |
| B6 | No retries / no timeouts. `tenacity` declared but unused; `per_file_timeout_s` never enforced; OpenAI/Reducto/LlamaExtract classify errors by string-matching `"429"` instead of SDK exception types | providers, runner | `tenacity.retry` on `ProviderTransientError` only; `concurrent.futures` timeout → `ProviderTimeoutError`; use `openai.RateLimitError`, `anthropic.RateLimitError` (already), etc. |
| B7 | `_images` input mode unimplemented: `pdf_to_images` exists, runner only fills `req.text`, providers only read `req.text`. Roster has `_text` pipelines only | plan D-06 | Fine to defer until D-06 is decided, but say so in README |
| B8 | `precomputed` is registered as a *public pipeline* with `results_dir=""` | `pipelines.py` | Remove from roster; construct via `resume-bench run precomputed --results-dir …` |
| B9 | Mini split is 1 resume with no PDF (`pdfs/mini_001.pdf` missing) → `resume-bench run … --split mini` crashes on `fitz.open`. Plan: 5 synthetic resumes with PDF+DOCX | `tests/fixtures/mini/` | Generate 5 small synthetic PDFs (LaTeX or reportlab) + GT; the `.gitignore` exception already exists |
| B10 | "Golden" fixture is a prediction identical to GT — it tests nothing. Plan B-04 = freeze *monolith* outputs on ~40 edge cases | `tests/fixtures/golden/` | Run `evaluation/metrics.py` from the monolith on curated pairs and store `{gt, pred, expected}`; parametrize a test over them |
| B11 | Report lacks failures.csv, latency/cost, difficulty breakdown, completed-only column, `_metadata.json` (commit, threshold, versions); `graded` shown instead of completion rate | `report/leaderboard.py` | plan §C.8 |

---

## C. Smaller issues

1. `.env.example` prefixes vendor keys (`RESUME_BENCH_OPENAI_API_KEY`); everyone already has `OPENAI_API_KEY`. Use pydantic `AliasChoices("OPENAI_API_KEY", "RESUME_BENCH_OPENAI_API_KEY")`.
2. `settings = Settings()` at import time; fine for CLI, awkward for tests. Add `get_settings()` with `lru_cache` and let tests override.
3. `pdf_to_images` uses `tempfile.mktemp` (deprecated, race-prone) and never deletes files. Use a `TemporaryDirectory` owned by the runner.
4. `download_dataset(split=…)` ignores `split`; use `allow_patterns` for `dev.jsonl`/`test.jsonl` + their file dirs.
5. `get_dataset_status` opens files without closing; `load_split` doesn't check `pdf_path.exists()`.
6. `validate_dataset` only validates `ground_truth`; also validate the record envelope (`files.pdf` present and exists, `difficulty` ∈ {easy,medium,hard}).
7. `grade_pipelines` writes `grades/*.grade.json` but no per-pipeline `_grade.json` summary; `report` re-aggregates from per-resume files — fine, but the aggregate should be written once by `grade`.
8. `CITATION.cff`: `type: dataset` → `software` (the HF dataset gets its own card); `website: careerflow.ai` → **careerflow.co**; `date-released` placeholder.
9. README pipeline table omits `openai_gpt-5.6-sol_text`; no "how to add a provider", no methodology link; `pip install -e ".[dev]"` should be `uv sync --extra dev`.
10. `.venv` is Python **3.14** — tests never ran on 3.10. Add `.python-version` = 3.10 and CI matrix (B-01 says 3.10 + 3.13).
11. Missing entirely: `uv.lock`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml` (gitleaks), `docs/*.md` (dirs are empty and untracked), `CONTRIBUTING.md`, `.claude/commands/integrate-provider.md`.
12. `hypothesis`/`respx` declared, unused. Either add the property tests (F1 ∈ [0,1], `score(x,x)=1`, permutation invariance) or drop the deps for now.
13. `providers/__init__.py` does not import the provider modules, so `@register_provider` never runs unless something imports `resume_bench.providers.openai` explicitly. `create_provider` only calls `load_plugins()` (entry points) on a miss — built-ins would fail with "no provider 'openai'". Import them in `__init__.py` (the plan says so).
14. `score_singleton` treats `gt` empty / `pred` filled as F1 0.5 (half the fields "match" as empty-vs-empty). Decide: hallucinated basics should score 0 on the filled fields, which it does — but the empty-vs-empty fields inflating to 1.0 is the same vacuous problem at field level. Exclude fields empty on both sides from the average.
15. Leaderboard HTML/CSV rank by penalized F1 only after A2 is fixed; today a pipeline that fails on 90% of resumes can rank first.

---

## D. What is good and should stay

- Package layout is exactly plan §C.2 minus the not-yet-built pieces; naming is consistent.
- `PipelineSpec` / `Provider` / `register_provider` / entry-point loading mirror ExtractBench cleanly and are small.
- `SectionSpec` + `SectionKind` is the right encoding of `SECTION_CONFIG`.
- Vacuous-section flag on `SectionScore` and `non_vacuous_sections` on `ResumeScore` implement the v1.1 metric fix correctly (apart from B1).
- Typer CLI surface matches §C.9; lazy imports inside commands keep `--help` fast.
- Boolean handling for `hasPersonalPhoto` and case-insensitive skill dedup are improvements over the monolith.
- Metric unit tests cover the edge cases that matter (empty sides, threshold, unequal lengths).

---

## E. Suggested order of fixes (maps to ACTION_PLAN tickets)

1. A3 + A4 — regenerate schema/prompt mechanically, add hash tests (B-02). 0.5 pd
2. A1 + B2 — `RunRecord.output`, `canonical.py`, grader reads `output` (B-03, B-11). 1 pd
3. A2 + B1 + B3 — failure policy, basics out of headline, summary kind (B-06, B-12). 1 pd
4. A5 + C13 + B6 — port monolith provider bodies, import providers in `__init__`, retries/timeouts (B-09, B-11). 1 pd
5. B9 + B10 — real mini split with PDFs; golden fixtures from the monolith (B-04, B-07). 1 pd
6. B5, B8, B11, C-items — cache metadata, roster cleanup, report columns, CI/lock/pre-commit (B-13, B-17). 1–1.5 pd

Then run the week-6 parity gate (G-01) early on `batch_009` — it would have caught A1–A4 immediately.

---

*Housekeeping:* a temporary archive I made for this review could not be deleted from your machine and was moved to `_to_delete/.reb_snapshot.tgz` in the repo folder — please delete that folder (it is untracked).

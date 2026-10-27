from __future__ import annotations

import json
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.schema.sections import get_sections
from resume_bench.settings import settings


def _load_pipeline_results(pipeline_name: str, split: str) -> dict[str, dict]:
    """Load all result files for a pipeline/split combination."""
    results_dir = settings.output_dir / pipeline_name / split
    results = {}

    if not results_dir.exists():
        return results

    for path in results_dir.glob("*.result.json"):
        with open(path) as f:
            record = json.load(f)

        resume_id = record.get("resume_id", path.stem.replace(".result", ""))

        if record.get("error"):
            continue

        canonical = record.get("output") or record.get("raw_output")
        if canonical:
            results[resume_id] = canonical

    return results


def grade_single_strict(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    verbatim: bool = False,
) -> dict:
    """Grade a single resume using ResumeBench-Strict scoring.

    Returns a dict with ``f1``, ``precision``, ``recall`` (section-averaged),
    ``correct``, ``expected``, ``predicted``, and ``sections``.
    """
    from resume_bench.grading.strict import score_resume_strict

    return score_resume_strict(ground_truth, prediction, get_sections(), verbatim=verbatim)


def grade_pipelines_strict(
    pipeline_names: list[str],
    split: str = "test",
    verbatim: bool = False,
) -> dict[str, dict[str, Any]]:
    """Grade multiple pipelines using ResumeBench-Strict scoring."""
    from resume_bench.grading.strict import score_resume_strict

    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}
    sections = get_sections()

    reports = {}

    for name in pipeline_names:
        predictions = _load_pipeline_results(name, split)

        scores: list[dict] = []
        errors = 0

        for resume_id, gt in gt_by_id.items():
            pred = predictions.get(resume_id)

            if pred is None:
                errors += 1
                scores.append({"resume_id": resume_id, "completed": False, "f1": 0.0})
                continue

            result = score_resume_strict(gt, pred, sections, verbatim=verbatim)
            result["resume_id"] = resume_id
            result["completed"] = True
            scores.append(result)

        completed = [s for s in scores if s.get("completed")]
        completed_count = len(completed)

        resume_f1s = [s["f1"] for s in completed]
        macro_f1 = sum(resume_f1s) / len(resume_f1s) if resume_f1s else 0.0

        total_correct = sum(s.get("correct", 0) for s in completed)
        total_expected = sum(s.get("expected", 0) for s in completed)
        total_predicted = sum(s.get("predicted", 0) for s in completed)

        grades_dir = settings.output_dir / name / split / "grades_strict"
        grades_dir.mkdir(parents=True, exist_ok=True)

        for s in scores:
            grade_path = grades_dir / f"{s['resume_id']}.grade.json"
            with open(grade_path, "w") as f:
                json.dump(s, f, indent=2)

        reports[name] = {
            "macro_f1": round(macro_f1, 4),
            "total_units": {
                "correct": total_correct,
                "expected": total_expected,
                "predicted": total_predicted,
            },
            "total_resumes": len(gt_by_id),
            "completed": completed_count,
            "errors": errors,
        }

    return reports

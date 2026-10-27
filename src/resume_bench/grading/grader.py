from __future__ import annotations

import json
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.grading.models import CellCounts, ResumeBenchScore
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


def grade_single_exact(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    exclude_sections: set[str] | None = None,
) -> ResumeBenchScore:
    """Grade a single resume using ResumeBench-exact scoring."""
    from resume_bench.grading.exact import rb_score_resume

    cells = rb_score_resume(ground_truth, prediction, get_sections(), exclude_sections)
    return ResumeBenchScore(cells=cells)


def grade_pipelines_exact(
    pipeline_names: list[str],
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    """Grade multiple pipelines using ResumeBench-exact cell-level scoring."""
    from resume_bench.grading.exact import rb_score_resume

    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}
    sections = get_sections()

    reports = {}

    for name in pipeline_names:
        predictions = _load_pipeline_results(name, split)

        scores: list[ResumeBenchScore] = []
        errors = 0

        for resume_id, gt in gt_by_id.items():
            pred = predictions.get(resume_id)

            if pred is None:
                errors += 1
                scores.append(ResumeBenchScore(resume_id=resume_id, completed=False))
                continue

            cells = rb_score_resume(gt, pred, sections)
            scores.append(ResumeBenchScore(resume_id=resume_id, cells=cells))

        total_correct = sum(s.cells.correct for s in scores if s.completed)
        total_expected = sum(s.cells.expected for s in scores if s.completed)
        total_predicted = sum(s.cells.predicted for s in scores if s.completed)

        agg = CellCounts(
            correct=total_correct,
            expected=total_expected,
            predicted=total_predicted,
        )

        completed_count = sum(1 for s in scores if s.completed)

        resume_f1s = [s.cells.f1 for s in scores if s.completed]
        avg_f1 = sum(resume_f1s) / len(resume_f1s) if resume_f1s else 0.0

        grades_dir = settings.output_dir / name / split / "grades_exact"
        grades_dir.mkdir(parents=True, exist_ok=True)

        for s in scores:
            grade_path = grades_dir / f"{s.resume_id}.grade.json"
            grade_data = {
                "resume_id": s.resume_id,
                "completed": s.completed,
                "correct": s.cells.correct,
                "expected": s.cells.expected,
                "predicted": s.cells.predicted,
                "precision": round(s.cells.precision, 4),
                "recall": round(s.cells.recall, 4),
                "f1": round(s.cells.f1, 4),
            }
            with open(grade_path, "w") as f:
                json.dump(grade_data, f, indent=2)

        reports[name] = {
            "micro_precision": round(agg.precision, 4),
            "micro_recall": round(agg.recall, 4),
            "micro_f1": round(agg.f1, 4),
            "macro_f1": round(avg_f1, 4),
            "total_cells": {
                "correct": total_correct,
                "expected": total_expected,
                "predicted": total_predicted,
            },
            "total_resumes": len(gt_by_id),
            "completed": completed_count,
            "errors": errors,
        }

    return reports


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

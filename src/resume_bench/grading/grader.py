from __future__ import annotations

import json
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.grading.metrics import SECTION_SCHEMA_FIELDS, score_entity_list, score_flat_list, score_singleton
from resume_bench.grading.models import ExtractBenchScore, GradingConfig, ResumeScore, SectionScore
from resume_bench.schema.sections import SectionKind, get_sections
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


def grade_single(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    cfg: GradingConfig = GradingConfig(),
) -> ResumeScore:
    """Grade a single resume prediction against ground truth."""
    score = ResumeScore()

    for spec in get_sections():
        gt_data = ground_truth.get(spec.name)
        pred_data = prediction.get(spec.name)

        if spec.kind == SectionKind.SINGLETON:
            gt_dict = gt_data if isinstance(gt_data, dict) else {}
            pred_dict = pred_data if isinstance(pred_data, dict) else {}

            section_score = score_singleton(gt_dict, pred_dict, spec.key_fields, cfg)

        elif spec.kind == SectionKind.FLAT_LIST:
            gt_list = gt_data if isinstance(gt_data, list) else []
            pred_list = pred_data if isinstance(pred_data, list) else []

            section_score = score_flat_list(gt_list, pred_list, cfg)

        elif spec.kind == SectionKind.ENTITY_LIST:
            if spec.name == "personalSummary":
                gt_text = gt_data if isinstance(gt_data, str) else ""
                pred_text = pred_data if isinstance(pred_data, str) else ""

                if not gt_text and not pred_text:
                    section_score = SectionScore(
                        precision=1.0, recall=1.0, f1=1.0, is_vacuous=True,
                    )
                elif not gt_text or not pred_text:
                    section_score = SectionScore(
                        gt_count=1 if gt_text else 0,
                        pred_count=1 if pred_text else 0,
                    )
                else:
                    from resume_bench.grading.text import edit_distance_ratio
                    f1 = edit_distance_ratio(gt_text, pred_text)
                    section_score = SectionScore(
                        gt_count=1,
                        pred_count=1,
                        precision=round(f1, 4),
                        recall=round(f1, 4),
                        f1=round(f1, 4),
                    )
            else:
                gt_list = gt_data if isinstance(gt_data, list) else []
                pred_list = pred_data if isinstance(pred_data, list) else []

                schema_fields = (
                    SECTION_SCHEMA_FIELDS.get(spec.name)
                    if cfg.score_all_fields else None
                )
                section_score = score_entity_list(
                    gt_list, pred_list, spec.key_fields,
                    score_description=spec.score_description,
                    schema_fields=schema_fields,
                    cfg=cfg,
                )
        else:
            continue

        score.sections[spec.name] = section_score

    return score


def grade_pipelines(
    pipeline_names: list[str],
    split: str = "test",
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Grade multiple pipelines and return summary reports."""
    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}

    cfg = GradingConfig(threshold=threshold)
    reports = {}

    for name in pipeline_names:
        predictions = _load_pipeline_results(name, split)

        scores = []
        errors = 0

        for resume_id, gt in gt_by_id.items():
            pred = predictions.get(resume_id)

            if pred is None:
                errors += 1

                failed_score = ResumeScore(resume_id=resume_id, completed=False)
                for spec in get_sections():
                    failed_score.sections[spec.name] = SectionScore()
                scores.append(failed_score)
                continue

            resume_score = grade_single(gt, pred, cfg)
            resume_score.resume_id = resume_id
            scores.append(resume_score)

        all_f1s = [s.macro_entity_f1 for s in scores]
        avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

        completed_f1s = [s.macro_entity_f1 for s in scores if s.completed]
        avg_completed_f1 = sum(completed_f1s) / len(completed_f1s) if completed_f1s else 0.0

        section_f1s = {}
        for spec in get_sections():
            vals = [
                s.sections[spec.name].f1
                for s in scores
                if spec.name in s.sections and not s.sections[spec.name].is_vacuous
            ]
            if vals:
                section_f1s[spec.name] = round(sum(vals) / len(vals), 4)

        grades_dir = settings.output_dir / name / split / "grades"
        grades_dir.mkdir(parents=True, exist_ok=True)

        for s in scores:
            grade_path = grades_dir / f"{s.resume_id}.grade.json"
            grade_data = {
                "resume_id": s.resume_id,
                "macro_entity_f1": round(s.macro_entity_f1, 4),
                "sections": {
                    k: {
                        "f1": v.f1,
                        "precision": v.precision,
                        "recall": v.recall,
                        "omission_rate": v.omission_rate,
                        "hallucination_rate": v.hallucination_rate,
                        "field_accuracy": v.field_accuracy,
                        "description_token_f1": v.description_token_f1,
                        "is_vacuous": v.is_vacuous,
                    }
                    for k, v in s.sections.items()
                },
            }
            with open(grade_path, "w") as f:
                json.dump(grade_data, f, indent=2)

        completed_count = sum(1 for s in scores if s.completed)

        reports[name] = {
            "resume_entity_f1": round(avg_f1, 4),
            "completed_only_f1": round(avg_completed_f1, 4),
            "section_f1": section_f1s,
            "total_resumes": len(gt_by_id),
            "completed": completed_count,
            "errors": errors,
            "completion_rate": completed_count / len(gt_by_id) if gt_by_id else 0.0,
        }

    return reports


def grade_single_extractbench(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    exclude_sections: set[str] | None = None,
) -> ExtractBenchScore:
    """Grade a single resume using ExtractBench-exact scoring."""
    from resume_bench.grading.extractbench import eb_score_resume

    cells = eb_score_resume(ground_truth, prediction, get_sections(), exclude_sections)
    return ExtractBenchScore(cells=cells)


def grade_pipelines_extractbench(
    pipeline_names: list[str],
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    """Grade multiple pipelines using ExtractBench-exact cell-level scoring."""
    from resume_bench.grading.extractbench import eb_score_resume

    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}
    sections = get_sections()

    reports = {}

    for name in pipeline_names:
        predictions = _load_pipeline_results(name, split)

        scores: list[ExtractBenchScore] = []
        errors = 0

        for resume_id, gt in gt_by_id.items():
            pred = predictions.get(resume_id)

            if pred is None:
                errors += 1
                scores.append(ExtractBenchScore(resume_id=resume_id, completed=False))
                continue

            cells = eb_score_resume(gt, pred, sections)
            scores.append(ExtractBenchScore(resume_id=resume_id, cells=cells))

        # Aggregate cell counts across all resumes
        total_correct = sum(s.cells.correct for s in scores if s.completed)
        total_expected = sum(s.cells.expected for s in scores if s.completed)
        total_predicted = sum(s.cells.predicted for s in scores if s.completed)

        from resume_bench.grading.models import CellCounts

        agg = CellCounts(
            correct=total_correct,
            expected=total_expected,
            predicted=total_predicted,
        )

        completed_count = sum(1 for s in scores if s.completed)

        # Per-resume F1 average
        resume_f1s = [s.cells.f1 for s in scores if s.completed]
        avg_f1 = sum(resume_f1s) / len(resume_f1s) if resume_f1s else 0.0

        grades_dir = settings.output_dir / name / split / "grades_extractbench"
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

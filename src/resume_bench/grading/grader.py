from __future__ import annotations

import json
from typing import Any

from resume_bench.dataset.loader import load_split
from resume_bench.grading.metrics import score_entity_list, score_flat_list, score_singleton
from resume_bench.grading.models import GradingConfig, ResumeScore, SectionScore
from resume_bench.schema.sections import SECTIONS, SectionKind
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

    for spec in SECTIONS:
        gt_data = ground_truth.get(spec.name)
        pred_data = prediction.get(spec.name)

        if spec.kind == SectionKind.SINGLETON:
            gt_dict = gt_data if isinstance(gt_data, dict) else {}
            pred_dict = pred_data if isinstance(pred_data, dict) else {}

            section_score = score_singleton(gt_dict, pred_dict, spec.key_fields, cfg)
            section_score.in_headline = False

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
                    from resume_bench.grading.text import token_f1
                    f1 = token_f1(gt_text, pred_text)
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

                section_score = score_entity_list(
                    gt_list, pred_list, spec.key_fields,
                    score_description=spec.score_description, cfg=cfg,
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
                for spec in SECTIONS:
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
        for spec in SECTIONS:
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

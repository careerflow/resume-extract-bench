from __future__ import annotations

from typing import Sequence

from resume_bench.grading.alignment import align_entities
from resume_bench.grading.models import GradingConfig, SectionScore
from resume_bench.grading.text import field_similarity, token_f1


def score_singleton(
    gt: dict,
    pred: dict,
    key_fields: Sequence[str],
    cfg: GradingConfig = GradingConfig(),
) -> SectionScore:
    """Score a singleton section (basics) - per-field accuracy."""
    field_scores = []
    field_acc = {}

    for f in key_fields:
        gt_val = gt.get(f, "")
        pred_val = pred.get(f, "")

        if isinstance(gt_val, bool) or isinstance(pred_val, bool):
            gt_bool = bool(gt_val) if gt_val is not None else None
            pred_bool = bool(pred_val) if pred_val is not None else None

            if gt_bool is None and pred_bool is None:
                sim = 1.0
            elif gt_bool is None or pred_bool is None:
                sim = 0.0
            else:
                sim = 1.0 if gt_bool == pred_bool else 0.0
        elif not gt_val and not pred_val:
            sim = 1.0
        elif not gt_val or not pred_val:
            sim = 0.0
        else:
            sim = field_similarity(gt_val, pred_val)

        field_acc[f] = round(sim, 4)
        field_scores.append(sim)

    avg = round(sum(field_scores) / len(field_scores), 4) if field_scores else 0.0

    is_vacuous = all(
        not gt.get(f) and not pred.get(f)
        for f in key_fields
    )

    return SectionScore(
        gt_count=1,
        pred_count=1,
        precision=avg,
        recall=avg,
        f1=avg,
        field_accuracy=field_acc,
        is_vacuous=is_vacuous,
    )


def score_flat_list(
    gt_items: list[dict],
    pred_items: list[dict],
    cfg: GradingConfig = GradingConfig(),
) -> SectionScore:
    """Score a flat skill list - flatten categories, deduplicate, align by name."""
    gt_skills = []
    for group in gt_items:
        gt_skills.extend(group.get("skills", []))

    pred_skills = []
    for group in pred_items:
        pred_skills.extend(group.get("skills", []))

    gt_skills = list({s.lower().strip(): s for s in gt_skills}.values())
    pred_skills = list({s.lower().strip(): s for s in pred_skills}.values())

    if not gt_skills and not pred_skills:
        return SectionScore(precision=1.0, recall=1.0, f1=1.0, is_vacuous=True)

    if not gt_skills:
        return SectionScore(
            pred_count=len(pred_skills),
            hallucination_rate=1.0,
        )

    if not pred_skills:
        return SectionScore(
            gt_count=len(gt_skills),
            omission_rate=1.0,
        )

    gt_wrapped = [{"name": s} for s in gt_skills]
    pred_wrapped = [{"name": s} for s in pred_skills]

    matched, missed_gt, spurious_pred = align_entities(
        gt_wrapped, pred_wrapped, key_fields=["name"], threshold=cfg.threshold,
    )

    precision = len(matched) / len(pred_skills)
    recall = len(matched) / len(gt_skills)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return SectionScore(
        gt_count=len(gt_skills),
        pred_count=len(pred_skills),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        omission_rate=round(len(missed_gt) / len(gt_skills), 4),
        hallucination_rate=round(len(spurious_pred) / len(pred_skills), 4),
    )


def score_entity_list(
    gt_items: list[dict],
    pred_items: list[dict],
    key_fields: Sequence[str],
    score_description: bool = False,
    cfg: GradingConfig = GradingConfig(),
) -> SectionScore:
    """Score a list section (experience, education, etc.) using Hungarian alignment."""
    if not gt_items and not pred_items:
        return SectionScore(precision=1.0, recall=1.0, f1=1.0, is_vacuous=True)

    if not gt_items:
        return SectionScore(
            pred_count=len(pred_items),
            hallucination_rate=1.0,
        )

    if not pred_items:
        return SectionScore(
            gt_count=len(gt_items),
            omission_rate=1.0,
        )

    matched, missed_gt, spurious_pred = align_entities(
        gt_items, pred_items, key_fields, threshold=cfg.threshold,
    )

    precision = len(matched) / len(pred_items)
    recall = len(matched) / len(gt_items)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    field_acc = {}

    if matched:
        for fld in key_fields:
            sims = [
                field_similarity(
                    gt_items[gi].get(fld, ""),
                    pred_items[pi].get(fld, ""),
                )
                for gi, pi, _ in matched
            ]
            field_acc[fld] = round(sum(sims) / len(sims), 4)

    desc_f1 = None

    if score_description and matched:
        desc_scores = []

        for gi, pi, _ in matched:
            gt_desc = " ".join(gt_items[gi].get("description", []))
            pred_desc = " ".join(pred_items[pi].get("description", []))

            if gt_desc.strip():
                desc_scores.append(token_f1(gt_desc, pred_desc))

        if desc_scores:
            desc_f1 = round(sum(desc_scores) / len(desc_scores), 4)

    return SectionScore(
        gt_count=len(gt_items),
        pred_count=len(pred_items),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        omission_rate=round(len(missed_gt) / len(gt_items), 4),
        hallucination_rate=round(len(spurious_pred) / len(pred_items), 4),
        field_accuracy=field_acc,
        description_token_f1=desc_f1,
    )

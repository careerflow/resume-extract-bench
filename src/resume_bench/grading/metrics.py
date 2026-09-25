from __future__ import annotations

from typing import Sequence

from resume_bench.grading.alignment import align_entities
from resume_bench.grading.models import GradingConfig, SectionScore
from resume_bench.grading.text import edit_distance_ratio, field_similarity, token_f1

# String fields that contain prose/text blocks rather than short identity values.
# These are scored with edit_distance_ratio (order-sensitive character comparison)
# instead of Jaro-Winkler (prefix-weighted, designed for short names/titles).
_TEXT_BLOCK_FIELDS = {"summary", "roleDescription", "text"}

# All scalar fields per section type (excludes 'description' arrays which are
# handled separately by _positional_bullet_score).
SECTION_SCHEMA_FIELDS: dict[str, list[str]] = {
    "experience": [
        "company", "position", "startMonth", "startYear",
        "endMonth", "endYear", "currentlyWorkHere", "city", "state",
        "country", "isRemote", "roleDescription", "companyUrl",
    ],
    "education": [
        "institution", "area", "studyType", "score",
        "startMonth", "startYear", "endMonth", "endYear",
        "currentlyStudyHere", "city", "state", "country",
    ],
    "projects": [
        "name", "startYear", "endYear", "url", "companyName",
        "city", "state", "country", "startMonth", "endMonth", "inProgress",
    ],
    "certifications": ["name", "issuer", "date", "url"],
    "awards": ["title", "awarder", "date", "summary", "url"],
    "volunteering": [
        "organization", "position", "startYear", "endYear",
        "currentlyVolunteerHere", "startMonth", "endMonth",
        "city", "state", "country", "summary",
    ],
    "publications": ["name", "publisher", "date", "summary"],
    "languages": ["name"],
    "interests": ["name"],
    "profiles": ["network", "url"],
    "customSections": ["sectionTitle", "summary"],
}


def _is_empty(val) -> bool:
    """Check if a value is effectively empty/null."""
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, list) and not val:
        return True
    if isinstance(val, dict) and all(_is_empty(v) for v in val.values()):
        return True
    return False


def _positional_bullet_score(gt_list: list, pred_list: list) -> float:
    """Score description bullets positionally using edit distance ratio.

    Compares bullet[0] vs bullet[0], bullet[1] vs bullet[1], etc.
    Missing or extra bullets score 0.0.
    """
    n = max(len(gt_list), len(pred_list))
    if n == 0:
        return 1.0
    scores = []
    for i in range(n):
        gt_bullet = str(gt_list[i]).strip() if i < len(gt_list) else ""
        pred_bullet = str(pred_list[i]).strip() if i < len(pred_list) else ""
        scores.append(edit_distance_ratio(gt_bullet, pred_bullet))
    return sum(scores) / len(scores)


def _entity_quality(
    gt_entity: dict,
    pred_entity: dict,
    schema_fields: list[str] | None = None,
) -> float:
    """Compute extraction quality for a matched entity pair.

    When schema_fields is None (default): scores only fields where GT has data.
    When schema_fields is provided: scores ALL listed fields using ExtractBench
    rules -- null/null = 1.0, null/value = 0.0, value/null = 0.0.
    """
    scores: list[float] = []

    # Determine which fields to iterate
    if schema_fields is not None:
        fields_to_score = schema_fields
    else:
        fields_to_score = list(gt_entity.keys())

    for key in fields_to_score:
        gt_val = gt_entity.get(key)
        pred_val = pred_entity.get(key)

        gt_empty = _is_empty(gt_val)
        pred_empty = _is_empty(pred_val)

        if gt_empty and pred_empty:
            if schema_fields is not None:
                scores.append(1.0)
            continue

        if gt_empty and not pred_empty:
            if schema_fields is not None:
                scores.append(0.0)
            continue

        if not gt_empty and pred_empty:
            scores.append(0.0)
            continue

        # Both have values — compare by type
        if isinstance(gt_val, list):
            pred_list = pred_val if isinstance(pred_val, list) else []
            scores.append(_positional_bullet_score(gt_val, pred_list))
        elif isinstance(gt_val, dict):
            # Nested object (e.g. URL {href, label}) — score sub-fields
            pred_dict = pred_val if isinstance(pred_val, dict) else {}
            sub_scores = []
            for sub_key in gt_val:
                gt_sub = gt_val.get(sub_key)
                pred_sub = pred_dict.get(sub_key)
                if _is_empty(gt_sub) and _is_empty(pred_sub):
                    sub_scores.append(1.0)
                elif _is_empty(gt_sub) or _is_empty(pred_sub):
                    sub_scores.append(0.0)
                else:
                    sub_scores.append(field_similarity(str(gt_sub), str(pred_sub)))
            scores.append(sum(sub_scores) / len(sub_scores) if sub_scores else 1.0)
        elif isinstance(gt_val, bool):
            scores.append(1.0 if gt_val == bool(pred_val) else 0.0)
        elif isinstance(gt_val, (int, float)):
            scores.append(1.0 if gt_val == pred_val else 0.0)
        elif isinstance(gt_val, str):
            pred_str = str(pred_val or "")
            if key in _TEXT_BLOCK_FIELDS:
                scores.append(edit_distance_ratio(gt_val, pred_str))
            else:
                scores.append(field_similarity(gt_val, pred_str))

    # In schema_fields mode, also score description arrays since they are
    # not listed in SECTION_SCHEMA_FIELDS (handled separately).
    if schema_fields is not None:
        for desc_key in ("description", "descriptions"):
            gt_desc = gt_entity.get(desc_key)
            pred_desc = pred_entity.get(desc_key)
            gt_has = isinstance(gt_desc, list) and gt_desc
            pred_has = isinstance(pred_desc, list) and pred_desc
            if gt_has:
                pred_list = pred_desc if pred_has else []
                scores.append(_positional_bullet_score(gt_desc, pred_list))
            elif pred_has:
                scores.append(0.0)
            # both empty → skip (no description to score)

    return sum(scores) / len(scores) if scores else 1.0


def _join_name(d: dict) -> str:
    """Join fname + lname into a single name string for comparison."""
    fname = (d.get("fname") or "").strip()
    lname = (d.get("lname") or "").strip()
    return f"{fname} {lname}".strip()


def score_singleton(
    gt: dict,
    pred: dict,
    key_fields: Sequence[str],
    cfg: GradingConfig = GradingConfig(),
) -> SectionScore:
    """Score a singleton section (basics) - per-field accuracy."""
    field_scores = []
    field_acc = {}

    # Join fname + lname into a single "name" comparison instead of scoring
    # them separately. This avoids penalising models that split the name at a
    # different boundary (e.g. "Jean Marie" / "Schiraldi" vs "Jean" / "Marie
    # Schiraldi"). Same idea as joining description bullets before token_f1.
    has_name_fields = "fname" in key_fields and "lname" in key_fields
    scored_fields = [f for f in key_fields if f not in ("fname", "lname")] if has_name_fields else list(key_fields)

    if has_name_fields:
        gt_name = _join_name(gt)
        pred_name = _join_name(pred)

        if not gt_name and not pred_name:
            sim = 1.0
        elif not gt_name or not pred_name:
            sim = 0.0
        else:
            sim = field_similarity(gt_name, pred_name)

        field_acc["name"] = round(sim, 4)
        field_scores.append(sim)

    for f in scored_fields:
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

    # Check vacuous using the effective fields (name instead of fname/lname)
    effective_fields = (["name"] if has_name_fields else []) + scored_fields
    is_vacuous = all(
        not (gt.get(f) or pred.get(f)) if f != "name"
        else not _join_name(gt) and not _join_name(pred)
        for f in effective_fields
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

    qualities = [sim for _, _, sim in matched]
    precision = sum(qualities) / len(pred_skills)
    recall = sum(qualities) / len(gt_skills)
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
    schema_fields: list[str] | None = None,
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

    qualities = [
        _entity_quality(gt_items[gi], pred_items[pi], schema_fields=schema_fields)
        for gi, pi, _ in matched
    ]
    precision = sum(qualities) / len(pred_items)
    recall = sum(qualities) / len(gt_items)
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
            gt_desc = gt_items[gi].get("description", [])
            pred_desc = pred_items[pi].get("description", [])

            if gt_desc:
                desc_scores.append(_positional_bullet_score(gt_desc, pred_desc))

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

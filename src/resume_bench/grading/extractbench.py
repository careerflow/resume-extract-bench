"""ExtractBench-exact scoring algorithm.

Cell-level binary exact-match scoring: every schema field is a "cell",
each cell is either correct (1) or incorrect (0) after whitespace
normalization.  Entity alignment uses the Hungarian algorithm on
cell-mismatch cost (no similarity threshold).

Ported from the internal ``run_combined_benchmark.py`` implementation.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from resume_bench.grading.metrics import SECTION_SCHEMA_FIELDS, _is_empty
from resume_bench.grading.models import CellCounts
from resume_bench.schema.sections import SectionKind, SectionSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eb_normalize(s: object) -> str:
    """Whitespace-collapse + lowercase + strip."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _eb_str_match(a: object, b: object) -> float:
    """Binary exact string comparison after normalization. Returns 1.0 or 0.0."""
    return 1.0 if _eb_normalize(a) == _eb_normalize(b) else 0.0


# ---------------------------------------------------------------------------
# Per-entity helpers
# ---------------------------------------------------------------------------

def _eb_cell_mismatch_count(
    gt_ent: dict,
    pred_ent: dict,
    s_fields: Sequence[str],
) -> tuple[int, int]:
    """Count mismatched cells between two entities (for Hungarian cost matrix).

    Each scalar field is one cell.  Each description bullet is one cell.
    Returns ``(mismatches, total)``.
    """
    mismatches = 0
    total = 0

    for field in s_fields:
        gt_v = gt_ent.get(field)
        pred_v = pred_ent.get(field)
        gt_e = _is_empty(gt_v)
        pred_e = _is_empty(pred_v)
        total += 1

        if gt_e and pred_e:
            continue  # match (both null)
        elif gt_e or pred_e:
            mismatches += 1
        elif isinstance(gt_v, bool):
            if gt_v != bool(pred_v):
                mismatches += 1
        elif isinstance(gt_v, (int, float)):
            if gt_v != pred_v:
                mismatches += 1
        elif isinstance(gt_v, str):
            if _eb_str_match(gt_v, str(pred_v or "")) == 0.0:
                mismatches += 1

    # Description bullets: each bullet is a cell
    for dk in ("description", "descriptions"):
        gt_d = gt_ent.get(dk, []) or []
        pred_d = pred_ent.get(dk, []) or []
        if not isinstance(gt_d, list):
            gt_d = []
        if not isinstance(pred_d, list):
            pred_d = []
        n = max(len(gt_d), len(pred_d))
        for i in range(n):
            total += 1
            gt_b = str(gt_d[i]).strip() if i < len(gt_d) else ""
            pred_b = str(pred_d[i]).strip() if i < len(pred_d) else ""
            if _eb_str_match(gt_b, pred_b) == 0.0:
                mismatches += 1

    return mismatches, total


def _eb_align_entities(
    gt_items: list[dict],
    pred_items: list[dict],
    s_fields: Sequence[str],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Align entities using Hungarian algorithm on cell-mismatch cost.

    Unlike the fuzzy aligner, there is **no** similarity threshold -- all
    assignments from the Hungarian solution are kept.

    Returns ``(matched_pairs, missed_gt_indices, spurious_pred_indices)``.
    """
    if not gt_items or not pred_items:
        return [], list(range(len(gt_items))), list(range(len(pred_items)))

    n, m = len(gt_items), len(pred_items)
    cost_matrix = np.zeros((n, m))

    for i, gt in enumerate(gt_items):
        for j, pred in enumerate(pred_items):
            mismatches, _ = _eb_cell_mismatch_count(gt, pred, s_fields)
            cost_matrix[i][j] = mismatches

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched: list[tuple[int, int]] = []
    missed_gt = set(range(n))
    spurious_pred = set(range(m))

    for r, c in zip(row_ind, col_ind):
        matched.append((r, c))
        missed_gt.discard(r)
        spurious_pred.discard(c)

    return matched, sorted(missed_gt), sorted(spurious_pred)


def _eb_count_entity_cells(entity: dict, s_fields: Sequence[str]) -> int:
    """Count total cells in an entity (scalar fields + description bullets)."""
    count = len(s_fields)
    for dk in ("description", "descriptions"):
        d = entity.get(dk, []) or []
        if isinstance(d, list):
            count += len(d)
    return max(count, 1)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def eb_score_resume(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
    exclude_sections: set[str] | None = None,
) -> CellCounts:
    """Score a resume using the ExtractBench-exact algorithm.

    Every schema field is a "cell".  A cell is **correct** when the GT and
    predicted values are identical after whitespace normalisation.

    Returns :class:`CellCounts` so the caller can compute P/R/F1.
    """
    correct = 0
    expected = 0
    predicted = 0
    exclude = set(exclude_sections or [])

    for spec in sections:
        if spec.name in exclude:
            continue

        # -- Singleton (basics) ------------------------------------------------
        if spec.kind == SectionKind.SINGLETON:
            gt_sec = gt.get(spec.name, {})
            pred_sec = pred.get(spec.name, {})
            if not isinstance(gt_sec, dict):
                gt_sec = {}
            if not isinstance(pred_sec, dict):
                pred_sec = {}
            fields = list(spec.key_fields)

            # fname + lname -> one "name" cell
            has_name = "fname" in fields and "lname" in fields
            if has_name:
                gt_name = (
                    (gt_sec.get("fname") or "") + " " + (gt_sec.get("lname") or "")
                ).strip()
                pred_name = (
                    (pred_sec.get("fname") or "") + " " + (pred_sec.get("lname") or "")
                ).strip()
                expected += 1
                predicted += 1
                if _eb_str_match(gt_name, pred_name) == 1.0:
                    correct += 1
                scored_fields = [f for f in fields if f not in ("fname", "lname")]
            else:
                scored_fields = fields

            for field in scored_fields:
                gt_v = gt_sec.get(field)
                pred_v = pred_sec.get(field)
                expected += 1
                predicted += 1
                if _is_empty(gt_v) and _is_empty(pred_v):
                    correct += 1
                elif _is_empty(gt_v) or _is_empty(pred_v):
                    pass  # mismatch
                elif isinstance(gt_v, bool) or isinstance(pred_v, bool):
                    if bool(gt_v) == bool(pred_v):
                        correct += 1
                else:
                    if _eb_str_match(str(gt_v or ""), str(pred_v or "")) == 1.0:
                        correct += 1

        # -- Flat skills -------------------------------------------------------
        elif spec.kind == SectionKind.FLAT_LIST:
            gt_items = gt.get(spec.name, [])
            pred_items = pred.get(spec.name, [])
            if not isinstance(gt_items, list):
                gt_items = []
            if not isinstance(pred_items, list):
                pred_items = []
            gt_skills: list[str] = []
            pred_skills: list[str] = []
            for g in gt_items:
                gt_skills.extend(g.get("skills", []))
            for p in pred_items:
                pred_skills.extend(p.get("skills", []))
            gt_skills = list({s.lower().strip(): s for s in gt_skills}.values())
            pred_skills = list({s.lower().strip(): s for s in pred_skills}.values())
            if not gt_skills and not pred_skills:
                continue

            # Align skills using exact match (Hungarian on mismatch cost)
            gt_w = [{"name": s} for s in gt_skills]
            pred_w = [{"name": s} for s in pred_skills]
            matched, missed, spurious = _eb_align_entities(gt_w, pred_w, ["name"])

            for gi, pi in matched:
                expected += 1
                predicted += 1
                if _eb_str_match(gt_skills[gi], pred_skills[pi]) == 1.0:
                    correct += 1
            expected += len(missed)    # missed GT skills
            predicted += len(spurious)  # hallucinated skills

        # -- Entity-list sections (+ personalSummary special case) -------------
        elif spec.kind == SectionKind.ENTITY_LIST:

            # personalSummary: stored as a plain string in the open-source schema
            # (internal repo wraps it as [{"text": "..."}] and uses entity-list path).
            # Match the internal repo's semantics: only count the side that has data.
            if spec.name == "personalSummary":
                gt_text = gt.get(spec.name)
                pred_text = pred.get(spec.name)
                gt_str = gt_text if isinstance(gt_text, str) else ""
                pred_str = pred_text if isinstance(pred_text, str) else ""
                gt_empty = not gt_str.strip()
                pred_empty = not pred_str.strip()
                if gt_empty and pred_empty:
                    continue
                if gt_empty:
                    # Spurious prediction — cells count toward predicted only
                    predicted += 1
                elif pred_empty:
                    # Missed GT — cells count toward expected only
                    expected += 1
                else:
                    # Both present — one cell, score it
                    expected += 1
                    predicted += 1
                    if _eb_str_match(gt_str, pred_str) == 1.0:
                        correct += 1
                continue

            gt_items = gt.get(spec.name, [])
            pred_items = pred.get(spec.name, [])
            if not isinstance(gt_items, list):
                gt_items = []
            if not isinstance(pred_items, list):
                pred_items = []
            if not gt_items and not pred_items:
                continue

            s_fields = SECTION_SCHEMA_FIELDS.get(spec.name, [])

            if not gt_items:
                # All predictions are spurious
                for pi in range(len(pred_items)):
                    predicted += _eb_count_entity_cells(pred_items[pi], s_fields)
                continue
            if not pred_items:
                # All GT is missed
                for gi in range(len(gt_items)):
                    expected += _eb_count_entity_cells(gt_items[gi], s_fields)
                continue

            matched, missed_gt, spurious_pred = _eb_align_entities(
                gt_items, pred_items, s_fields,
            )

            # Matched pairs: score each cell
            for gi, pi in matched:
                gt_ent = gt_items[gi]
                pred_ent = pred_items[pi]

                for field in s_fields:
                    gt_v = gt_ent.get(field)
                    pred_v = pred_ent.get(field)
                    gt_e = _is_empty(gt_v)
                    pred_e = _is_empty(pred_v)
                    expected += 1
                    predicted += 1

                    if gt_e and pred_e:
                        correct += 1
                    elif gt_e or pred_e:
                        pass  # mismatch
                    elif isinstance(gt_v, bool):
                        if gt_v == bool(pred_v):
                            correct += 1
                    elif isinstance(gt_v, (int, float)):
                        if gt_v == pred_v:
                            correct += 1
                    elif isinstance(gt_v, str):
                        if _eb_str_match(gt_v, str(pred_v or "")) == 1.0:
                            correct += 1

                # Description bullets: each bullet is a cell
                for dk in ("description", "descriptions"):
                    gt_d = gt_ent.get(dk, []) or []
                    pred_d = pred_ent.get(dk, []) or []
                    if not isinstance(gt_d, list):
                        gt_d = []
                    if not isinstance(pred_d, list):
                        pred_d = []
                    n_bullets = max(len(gt_d), len(pred_d))
                    for i in range(n_bullets):
                        gt_b = str(gt_d[i]).strip() if i < len(gt_d) else ""
                        pred_b = str(pred_d[i]).strip() if i < len(pred_d) else ""
                        if i < len(gt_d):
                            expected += 1
                        if i < len(pred_d):
                            predicted += 1
                        if _eb_str_match(gt_b, pred_b) == 1.0:
                            correct += 1

            # Missed GT entities: cells count toward expected only
            for gi in missed_gt:
                expected += _eb_count_entity_cells(gt_items[gi], s_fields)

            # Spurious pred entities: cells count toward predicted only
            for pi in spurious_pred:
                predicted += _eb_count_entity_cells(pred_items[pi], s_fields)

    return CellCounts(correct=correct, expected=expected, predicted=predicted)

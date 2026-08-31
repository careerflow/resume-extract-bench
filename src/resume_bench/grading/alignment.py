from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from resume_bench.grading.text import field_similarity


def align_entities(
    gt_items: list[dict],
    pred_items: list[dict],
    key_fields: Sequence[str],
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Hungarian alignment on mean Jaro-Winkler over key_fields.

    Returns (matched_pairs, missed_gt_indices, spurious_pred_indices).
    Each matched pair is (gt_idx, pred_idx, similarity).
    """
    if not gt_items or not pred_items:
        return [], list(range(len(gt_items))), list(range(len(pred_items)))

    n, m = len(gt_items), len(pred_items)

    sim_matrix = np.zeros((n, m))

    for i, gt in enumerate(gt_items):
        for j, pred in enumerate(pred_items):
            field_sims = [
                field_similarity(gt.get(field, ""), pred.get(field, ""))
                for field in key_fields
            ]
            sim_matrix[i][j] = np.mean(field_sims)

    row_ind, col_ind = linear_sum_assignment(1 - sim_matrix)

    matched = []
    missed_gt = set(range(n))
    spurious_pred = set(range(m))

    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r][c] >= threshold:
            matched.append((r, c, float(sim_matrix[r][c])))
            missed_gt.discard(r)
            spurious_pred.discard(c)

    return matched, sorted(missed_gt), sorted(spurious_pred)

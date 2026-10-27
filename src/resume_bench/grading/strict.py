"""ResumeBench-Strict scoring algorithm.

Record-level binary scoring for structured resume extraction.  Three rules
make it stricter than per-cell scoring:

1. **Record-level units.**  An entity (one job, school, project, certification,
   award, volunteer role, publication, language, interest, profile, custom
   section, summary) is ONE unit: it is correct only if it is aligned AND every
   field that is non-empty on either side matches AND its bullets match
   one-for-one (same count, each bullet equal).
2. **Empty is not free.**  Basics fields count only where the label or the
   parser has a value.  Skills: one unit per skill.
3. **Section-averaged.**  Per document, F1 / precision / recall are computed per
   section (sections present in the label or the output) and averaged, so every
   section weighs the same.

Benchmark v3 adds a fourth rule with ``verbatim=True``:

4. **Verbatim text.**  Strings match only after whitespace collapsing -- case
   and punctuation count.  Typographic variants of one character are folded
   (hyphens/dashes -> "-", apostrophes -> "'", double quotes -> '"', soft
   hyphens dropped).

"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from resume_bench.schema.sections import SectionKind, SectionSpec

__all__ = ["score_resume_strict", "verbatim_normalize", "SECTION_SCHEMA_FIELDS"]

# ---------------------------------------------------------------------------
# Schema field definitions
# ---------------------------------------------------------------------------

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
    "languages": ["name", "level"],
    "interests": ["name"],
    "profiles": ["network", "url"],
    "customSections": ["sectionTitle", "summary"],
    "personalSummary": ["text"],
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, list) and not val:
        return True
    if isinstance(val, dict) and all(_is_empty(v) for v in val.values()):
        return True
    return False


def _unwrap_items(section_data):
    """Extract the items list from a possibly-wrapped section."""
    if isinstance(section_data, dict) and "items" in section_data:
        return section_data.get("items", [])
    if isinstance(section_data, list):
        return section_data
    return []


def _unwrap_text(section_data):
    """Extract the text string from a possibly-wrapped personalSummary."""
    if isinstance(section_data, dict) and "text" in section_data:
        return section_data.get("text")
    if isinstance(section_data, str):
        return section_data
    return None


def _rb_normalize(s: object) -> str:
    """Whitespace-collapse + lowercase + strip non-semantic punctuation.

    Keeps letters, digits, underscores (\\w), whitespace, '+', '#' so that
    C++, C#, F# survive.
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = re.sub(r"[^\w\s+#]", "", s)
    s = re.sub(r" +", " ", s).strip()
    return s


def _split_comma_skills(skills: list[str]) -> list[str]:
    """Split comma-separated skill strings into individual skills."""
    result: list[str] = []
    for s in skills:
        if not isinstance(s, str):
            result.append(s)
            continue
        if "," in s:
            result.extend(part.strip() for part in s.split(",") if part.strip())
        else:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Hungarian entity alignment (cell-mismatch cost, no threshold)
# ---------------------------------------------------------------------------

def _rb_cell_mismatch_count(
    gt_ent: dict,
    pred_ent: dict,
    s_fields: Sequence[str],
) -> tuple[int, int]:
    """Count mismatched cells between two entities (for Hungarian cost matrix)."""
    mismatches = 0
    total = 0

    for field in s_fields:
        gt_v = gt_ent.get(field)
        pred_v = pred_ent.get(field)
        gt_e = _is_empty(gt_v)
        pred_e = _is_empty(pred_v)
        total += 1

        if gt_e and pred_e:
            continue
        elif gt_e or pred_e:
            mismatches += 1
        elif isinstance(gt_v, bool):
            if gt_v != bool(pred_v):
                mismatches += 1
        elif isinstance(gt_v, (int, float)):
            if gt_v != pred_v:
                mismatches += 1
        elif isinstance(gt_v, str):
            if _rb_normalize(gt_v) != _rb_normalize(str(pred_v or "")):
                mismatches += 1

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
            if _rb_normalize(gt_b) != _rb_normalize(pred_b):
                mismatches += 1

    return mismatches, total


def _rb_align_entities(
    gt_items: list[dict],
    pred_items: list[dict],
    s_fields: Sequence[str],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Align entities using Hungarian algorithm on cell-mismatch cost.

    Returns ``(matched_pairs, missed_gt_indices, spurious_pred_indices)``.
    """
    if not gt_items or not pred_items:
        return [], list(range(len(gt_items))), list(range(len(pred_items)))

    n, m = len(gt_items), len(pred_items)
    cost_matrix = np.zeros((n, m))

    for i, gt in enumerate(gt_items):
        for j, pred in enumerate(pred_items):
            mismatches, _ = _rb_cell_mismatch_count(gt, pred, s_fields)
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


# ---------------------------------------------------------------------------
# Verbatim normalization + text-rule context manager
# ---------------------------------------------------------------------------

_GLYPHS = str.maketrans(
    {
        **{c: "-" for c in "‐‑‒–—―−"},
        **{c: "'" for c in "‘’‚‛′ʼ"},
        **{c: '"' for c in "“”„‟″"},
        "­": None,
    }
)

_active_normalize = _rb_normalize


def verbatim_normalize(s: object) -> str:
    """Whitespace collapsing + typographic variant folding; case and punctuation are kept."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).translate(_GLYPHS)).strip()


@contextmanager
def _text_rule(verbatim: bool):
    """Swap the string normalization for the duration of a scoring call."""
    global _active_normalize
    _active_normalize = verbatim_normalize if verbatim else _rb_normalize
    try:
        yield
    finally:
        _active_normalize = _rb_normalize


# ---------------------------------------------------------------------------
# Strict-specific helpers
# ---------------------------------------------------------------------------

def _str_match(a: object, b: object) -> float:
    """Binary exact string comparison after the active normalization."""
    return 1.0 if _active_normalize(a) == _active_normalize(b) else 0.0


def _prf(c: int, e: int, p: int) -> tuple[float, float, float]:
    pr = c / p if p > 0 else 1.0
    rc = c / e if e > 0 else 1.0
    return pr, rc, (2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0


def _empty(v: object) -> bool:
    return _is_empty(v) if not isinstance(v, str) else not v.strip()


def _same(gv: object, pv: object) -> bool:
    if isinstance(gv, bool) or isinstance(pv, bool):
        return bool(gv) == bool(pv)
    if isinstance(gv, (int, float)) and not isinstance(gv, bool):
        return gv == pv
    return _str_match(str(gv), str(pv if pv is not None else "")) == 1.0


def _bullets(ent: dict) -> list[str]:
    out: list[str] = []
    for dk in ("description", "descriptions"):
        d = ent.get(dk) or []
        if isinstance(d, list):
            out += [str(x).strip() for x in d]
    return out


def _record_ok(g: dict, p: dict, fields: list[str]) -> bool:
    """True if the two aligned entities match on every non-empty field and every bullet."""
    for f in fields:
        gv, pv = g.get(f), p.get(f)
        if _empty(gv) and _empty(pv):
            continue
        if _empty(gv) or _empty(pv) or not _same(gv, pv):
            return False
    gb, pb = _bullets(g), _bullets(p)
    return len(gb) == len(pb) and all(_str_match(a, b) == 1.0 for a, b in zip(gb, pb))


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_resume_strict(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
    verbatim: bool = False,
) -> dict:
    """Score a single resume with ResumeBench-Strict rules.

    Returns a dict with top-level ``f1``, ``precision``, ``recall`` (section-
    averaged), ``correct``, ``expected``, ``predicted`` (unit totals), and
    ``sections`` mapping section name -> per-section stats.
    """
    with _text_rule(verbatim):
        return _score_resume_strict(gt, pred, sections)


def _score_resume_strict(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
) -> dict:
    sec_results: dict[str, dict] = {}

    for spec in sections:
        name = spec.name

        if spec.kind == SectionKind.SINGLETON:
            g = gt.get(name, {}) or {}
            p = pred.get(name, {}) or {}
            if not isinstance(g, dict):
                g = {}
            if not isinstance(p, dict):
                p = {}
            fields = list(spec.key_fields)

            cells: list[tuple] = []
            if "fname" in fields and "lname" in fields:
                gn = ((g.get("fname") or "") + " " + (g.get("lname") or "")).strip()
                pn = ((p.get("fname") or "") + " " + (p.get("lname") or "")).strip()
                cells.append((gn, pn))
                fields = [f for f in fields if f not in ("fname", "lname")]
            cells += [(g.get(f), p.get(f)) for f in fields]

            c = e = q = 0
            for gv, pv in cells:
                ge, pe = _empty(gv), _empty(pv)
                e += not ge
                q += not pe
                c += (not ge and not pe and _same(gv, pv))

        elif spec.kind == SectionKind.FLAT_LIST:
            gs: list[str] = []
            ps: list[str] = []
            for x in _unwrap_items(gt.get(name, [])):
                gs.extend(x.get("skills", []))
            for x in _unwrap_items(pred.get(name, [])):
                ps.extend(x.get("skills", []))
            gs = list({s.lower().strip(): s for s in _split_comma_skills(gs)}.values())
            ps = list({s.lower().strip(): s for s in _split_comma_skills(ps)}.values())
            if not gs and not ps:
                continue
            m, _, _ = _rb_align_entities(
                [{"name": s} for s in gs], [{"name": s} for s in ps], ["name"],
            )
            c = sum(_str_match(gs[a], ps[b]) == 1.0 for a, b in m)
            e, q = len(gs), len(ps)

        elif spec.kind == SectionKind.ENTITY_LIST:
            if name == "personalSummary":
                gt_str = _unwrap_text(gt.get(name)) or ""
                pred_str = _unwrap_text(pred.get(name)) or ""
                gt_empty = not gt_str.strip()
                pred_empty = not pred_str.strip()
                if gt_empty and pred_empty:
                    continue
                c = int(not gt_empty and not pred_empty and _str_match(gt_str, pred_str) == 1.0)
                e = int(not gt_empty)
                q = int(not pred_empty)
            else:
                gi = _unwrap_items(gt.get(name, []))
                pi = _unwrap_items(pred.get(name, []))
                if not gi and not pi:
                    continue
                sf = SECTION_SCHEMA_FIELDS.get(name, [])
                c, e, q = 0, len(gi), len(pi)
                if gi and pi:
                    m, _, _ = _rb_align_entities(gi, pi, sf)
                    c = sum(_record_ok(gi[a], pi[b], sf) for a, b in m)
        else:
            continue

        if not e and not q:
            continue
        pr, rc, f1 = _prf(c, e, q)
        sec_results[name] = {
            "correct": c, "expected": e, "predicted": q,
            "f1": round(f1, 6), "precision": round(pr, 6), "recall": round(rc, 6),
        }

    n = len(sec_results)
    mean_fn = lambda k: sum(s[k] for s in sec_results.values()) / n if n else 1.0
    return {
        "f1": mean_fn("f1"),
        "precision": mean_fn("precision"),
        "recall": mean_fn("recall"),
        "correct": sum(s["correct"] for s in sec_results.values()),
        "expected": sum(s["expected"] for s in sec_results.values()),
        "predicted": sum(s["predicted"] for s in sec_results.values()),
        "sections": sec_results,
    }

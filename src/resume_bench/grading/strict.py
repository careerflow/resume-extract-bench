"""ResumeBench-Strict scoring algorithm.

Per-field scoring with direct identity-field matching for structured resume
extraction.

1. **Per-field units.**  Each field in a matched entity is scored
   independently — partial credit for records that match some fields but
   not all.
2. **Identity matching.**  Entities in structured sections (experience,
   education, projects, etc.) are matched by identity fields (company +
   position + years, institution + studyType + years, …) using direct
   greedy matching with a configurable gate (default 1.0 = all identity
   fields must match).  Keyword sections (languages, interests, profiles,
   publications) use Hungarian alignment.
3. **Empty is not free.**  Basics fields count only where the label or the
   parser has a value.  Skills: one unit per skill.
4. **Section-averaged.**  Per document, F1 / precision / recall are computed
   per section (sections present in the label or the output) and averaged,
   so every section weighs the same.
5. **Verbatim text.**  Strings match only after whitespace collapsing — case
   and punctuation count.  Typographic variants of one character are folded
   (hyphens/dashes → "-", apostrophes → "'", double quotes → '"', soft
   hyphens dropped).  Contacts (email, phone, linkedinUrl) and locations
   (city, state, country) are compared case-insensitively.
6. **Name composition.**  The ``basics`` name is composed from
   ``prefix + fname + lname + suffix``, whitespace-collapsed.

"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from resume_bench.schema.sections import SectionKind, SectionSpec

__all__ = [
    "score_resume_strict",
    "verbatim_normalize",
    "SECTION_SCHEMA_FIELDS",
    "IDENTITY_FIELDS",
    "KEYWORD_SECTIONS",
    "CONTACT_FIELDS",
    "LOCATION_FIELDS",
    "DEFAULT_GATE",
]

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
# Identity matching constants
# ---------------------------------------------------------------------------

DEFAULT_GATE = 1.0

CONTACT_FIELDS = {"email", "phone", "linkedinUrl"}
LOCATION_FIELDS = {"city", "state", "country"}

IDENTITY_FIELDS: dict[str, set[str]] = {
    "experience": {"company", "position", "startYear", "endYear"},
    "education": {"institution", "studyType", "startYear", "endYear"},
    "projects": {"name", "startYear", "endYear"},
    "certifications": {"title", "issuer"},
    "awards": {"title"},
    "volunteering": {"company", "position", "startYear", "endYear"},
    "customSections": {"sectionTitle"},
    "publications": {"name"},
    "languages": {"name"},
    "interests": {"name"},
    "profiles": {"network"},
}

KEYWORD_SECTIONS = {"languages", "interests", "profiles", "publications"}

_NAME_PARTS = ("prefix", "fname", "lname", "suffix")

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
    """Extract the items list from a possibly-wrapped section.

    - list → return as-is (already a list of entities)
    - dict with "items" key → return items list (standard wrapper)
    - dict without "items" → return [dict] (singleton like personalSummary)
    - anything else → return []
    """
    if isinstance(section_data, list):
        return section_data
    if isinstance(section_data, dict):
        if "items" in section_data:
            return section_data["items"]
        return [section_data]
    return []


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
            if _active_normalize(gt_v) != _active_normalize(str(pred_v or "")):
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
            if _active_normalize(gt_b) != _active_normalize(pred_b):
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
# Scoring helpers
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


def _bullets(ent: dict) -> list[str]:
    out: list[str] = []
    for dk in ("description", "descriptions"):
        d = ent.get(dk) or []
        if isinstance(d, list):
            out += [str(x).strip() for x in d]
    return out


def _field_passes(fname: str, gv: object, pv: object) -> bool:
    """Check if a single field passes its type-appropriate matching rule."""
    if isinstance(gv, bool) or isinstance(pv, bool):
        return bool(gv) == bool(pv)
    if isinstance(gv, (int, float)) and not isinstance(gv, bool):
        return gv == pv
    gs = str(gv if gv is not None else "")
    ps = str(pv if pv is not None else "")
    if fname in CONTACT_FIELDS or fname in LOCATION_FIELDS:
        return verbatim_normalize(gs).lower() == verbatim_normalize(ps).lower()
    return _str_match(gs, ps) == 1.0


def _identity_pass_rate(
    g: dict, p: dict, section_name: str,
) -> tuple[float, int]:
    """Fraction of non-empty identity fields that match between two entities."""
    id_fields = IDENTITY_FIELDS.get(section_name, set())
    if not id_fields:
        return 1.0, 0
    id_pass = id_total = 0
    for f in id_fields:
        gv, pv = g.get(f), p.get(f)
        if _empty(gv) and _empty(pv):
            continue
        id_total += 1
        if not _empty(gv) and not _empty(pv) and _field_passes(f, gv, pv):
            id_pass += 1
    if id_total == 0:
        return 1.0, 0
    return id_pass / id_total, id_total


def _direct_match(
    gi: list[dict],
    pi: list[dict],
    section_name: str,
    gate: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy direct matching by identity fields.

    Returns ``(matched, missed_gt, spurious_pred)``.
    """
    used_pred: set[int] = set()
    matched: list[tuple[int, int]] = []
    for g_idx, g_ent in enumerate(gi):
        best_p_idx = None
        best_rate = -1.0
        for p_idx, p_ent in enumerate(pi):
            if p_idx in used_pred:
                continue
            rate, _ = _identity_pass_rate(g_ent, p_ent, section_name)
            if rate >= gate and rate > best_rate:
                best_rate = rate
                best_p_idx = p_idx
        if best_p_idx is not None:
            matched.append((g_idx, best_p_idx))
            used_pred.add(best_p_idx)
    missed_gt = [i for i in range(len(gi)) if i not in {a for a, _ in matched}]
    spurious_pred = [i for i in range(len(pi)) if i not in used_pred]
    return matched, missed_gt, spurious_pred


def _score_record(
    g: dict, p: dict, fields: Sequence[str],
) -> tuple[int, int, int]:
    """Per-field scoring of a matched entity pair.  Returns ``(c, e, p)``."""
    c = e = q = 0
    for f in fields:
        gv, pv = g.get(f), p.get(f)
        ge, pe = _empty(gv), _empty(pv)
        if ge and pe:
            continue
        if not ge:
            e += 1
        if not pe:
            q += 1
        if not ge and not pe and _field_passes(f, gv, pv):
            c += 1

    gb, pb = _bullets(g), _bullets(p)
    for i in range(max(len(gb), len(pb))):
        gval = gb[i] if i < len(gb) else ""
        pval = pb[i] if i < len(pb) else ""
        g_empty = not gval.strip()
        p_empty = not pval.strip()
        if g_empty and p_empty:
            continue
        if not g_empty:
            e += 1
        if not p_empty:
            q += 1
        if not g_empty and not p_empty and _str_match(gval, pval) == 1.0:
            c += 1

    return c, e, q


def _cnt(ent: dict, sf: Sequence[str]) -> int:
    """Count non-empty fields in an unmatched entity."""
    n = 0
    for f in sf:
        if not _empty(ent.get(f)):
            n += 1
    for dk in ("description", "descriptions"):
        d = ent.get(dk, []) or []
        if isinstance(d, list):
            n += sum(1 for b in d if str(b).strip())
    return n


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_resume_strict(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
    gate: float = DEFAULT_GATE,
    verbatim: bool = True,
) -> dict:
    """Score a single resume with ResumeBench-Strict rules.

    Per-field scoring with direct identity matching for entity-list sections,
    Hungarian alignment for keyword sections and skills.

    Returns a dict with top-level ``f1``, ``precision``, ``recall`` (section-
    averaged), ``correct``, ``expected``, ``predicted`` (unit totals), and
    ``sections`` mapping section name to per-section stats.
    """
    with _text_rule(verbatim):
        return _score_resume_strict(gt, pred, sections, gate)


def _score_resume_strict(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
    gate: float,
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
            c = e = q = 0

            if "fname" in fields and "lname" in fields:
                gn = " ".join(g.get(k) or "" for k in _NAME_PARTS).strip()
                gn = " ".join(gn.split())
                pn = " ".join(p.get(k) or "" for k in _NAME_PARTS).strip()
                pn = " ".join(pn.split())
                ge, pe = _empty(gn), _empty(pn)
                if not (ge and pe):
                    if not ge:
                        e += 1
                    if not pe:
                        q += 1
                    if not ge and not pe and _str_match(gn, pn) == 1.0:
                        c += 1
                fields = [f for f in fields if f not in _NAME_PARTS]

            for f in fields:
                gv, pv = g.get(f), p.get(f)
                ge, pe = _empty(gv), _empty(pv)
                if ge and pe:
                    continue
                if not ge:
                    e += 1
                if not pe:
                    q += 1
                if not ge and not pe and _field_passes(f, gv, pv):
                    c += 1

        elif spec.kind == SectionKind.FLAT_LIST:
            gs_list: list[str] = []
            ps_list: list[str] = []
            for x in _unwrap_items(gt.get(name, [])):
                gs_list.extend(x.get("skills", []))
            for x in _unwrap_items(pred.get(name, [])):
                ps_list.extend(x.get("skills", []))
            gs_list = list(
                {s.lower().strip(): s for s in _split_comma_skills(gs_list)}.values()
            )
            ps_list = list(
                {s.lower().strip(): s for s in _split_comma_skills(ps_list)}.values()
            )
            if not gs_list and not ps_list:
                continue
            m, _, _ = _rb_align_entities(
                [{"name": s} for s in gs_list],
                [{"name": s} for s in ps_list],
                ["name"],
            )
            c = sum(_str_match(gs_list[a], ps_list[b]) == 1.0 for a, b in m)
            e, q = len(gs_list), len(ps_list)

        elif spec.kind == SectionKind.ENTITY_LIST:
            gi = _unwrap_items(gt.get(name, []))
            pi = _unwrap_items(pred.get(name, []))
            if not gi and not pi:
                continue
            sf = SECTION_SCHEMA_FIELDS.get(name, [])

            if name in KEYWORD_SECTIONS:
                mt, mgt, spr = _rb_align_entities(gi, pi, sf)
                c = e = q = 0
                for a, b in mt:
                    rc_, re_, rq = _score_record(gi[a], pi[b], sf)
                    c += rc_
                    e += re_
                    q += rq
                for idx in mgt:
                    e += _cnt(gi[idx], sf)
                for idx in spr:
                    q += _cnt(pi[idx], sf)
            else:
                c = e = q = 0
                if not gi:
                    for x in pi:
                        q += _cnt(x, sf)
                elif not pi:
                    for x in gi:
                        e += _cnt(x, sf)
                else:
                    matched, missed, spurious = _direct_match(
                        gi, pi, name, gate,
                    )
                    for a, b in matched:
                        rc_, re_, rq = _score_record(gi[a], pi[b], sf)
                        c += rc_
                        e += re_
                        q += rq
                    for idx in missed:
                        e += _cnt(gi[idx], sf)
                    for idx in spurious:
                        q += _cnt(pi[idx], sf)
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

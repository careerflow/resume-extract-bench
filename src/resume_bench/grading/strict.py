"""ExtractBench-Strict scoring algorithm.

A harder companion to ExtractBench-exact (:mod:`extractbench`).  Same inputs,
same normalization, same Hungarian entity alignment, same skills handling
(comma split + dedupe) and the same per-document averaging.  Three rules make
it stricter:

1. **Record-level units.**  An entity (one job, school, project, certification,
   award, volunteer role, publication, language, interest, profile, custom
   section, summary) is ONE unit: it is correct only if it is aligned AND every
   field that is non-empty on either side matches AND its bullets match
   one-for-one (same count, each bullet equal).
2. **Empty is not free.**  Basics fields count only where the label or the
   parser has a value.  Skills: one unit per skill, as in ExtractBench-exact.
3. **Section-averaged.**  Per document, F1 / precision / recall are computed per
   section (sections present in the label or the output) and averaged, so every
   section weighs the same.

Benchmark v3 adds a fourth rule with ``verbatim=True``:

4. **Verbatim text.**  Strings match only after whitespace collapsing — case
   and punctuation count.  Typographic variants of one character are folded
   (hyphens/dashes → "-", apostrophes → "'", double quotes → '"', soft hyphens
   dropped).

Ported from the internal ``scripts/strict_scoring.py`` implementation.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Sequence

from resume_bench.grading.extractbench import (
    _eb_align_entities,
    _eb_normalize,
    _split_comma_skills,
    _unwrap_items,
    _unwrap_text,
)
from resume_bench.grading.metrics import SECTION_SCHEMA_FIELDS, _is_empty
from resume_bench.schema.sections import SectionKind, SectionSpec

_OFFICIAL_NORMALIZE = _eb_normalize

_GLYPHS = str.maketrans(
    {
        **{c: "-" for c in "‐‑‒–—―−"},
        **{c: "'" for c in "‘’‚‛′ʼ"},
        **{c: '"' for c in "“”„‟″"},
        "­": None,
    }
)

# Module-level reference to the active normalize function.
# _text_rule swaps it for the duration of a scoring call.
_active_normalize = _OFFICIAL_NORMALIZE


def verbatim_normalize(s: object) -> str:
    """Whitespace collapsing + typographic variant folding; case and punctuation are kept."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).translate(_GLYPHS)).strip()


@contextmanager
def _text_rule(verbatim: bool):
    """Swap the string normalization for the duration of a scoring call."""
    global _active_normalize
    _active_normalize = verbatim_normalize if verbatim else _OFFICIAL_NORMALIZE
    try:
        yield
    finally:
        _active_normalize = _OFFICIAL_NORMALIZE


def _str_match(a: object, b: object) -> float:
    """Binary exact string comparison after the active normalization."""
    na = _active_normalize(a)
    nb = _active_normalize(b)
    return 1.0 if na == nb else 0.0


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


def score_resume_strict(
    gt: dict,
    pred: dict,
    sections: Sequence[SectionSpec],
    verbatim: bool = False,
) -> dict:
    """Score a single resume with ExtractBench-Strict rules.

    Returns a dict with top-level ``f1``, ``precision``, ``recall`` (section-
    averaged), ``correct``, ``expected``, ``predicted`` (unit totals), and
    ``sections`` mapping section name → per-section stats.
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
            m, _, _ = _eb_align_entities(
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
                    m, _, _ = _eb_align_entities(gi, pi, sf)
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

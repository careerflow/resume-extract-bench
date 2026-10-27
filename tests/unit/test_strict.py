"""Tests for ExtractBench-Strict scoring module."""

from resume_bench.grading.strict import (
    _record_ok,
    score_resume_strict,
    verbatim_normalize,
)
from resume_bench.schema.sections import SectionKind, SectionSpec


# ---------------------------------------------------------------------------
# verbatim_normalize
# ---------------------------------------------------------------------------


class TestVerbatimNormalize:

    def test_whitespace_collapsing(self):
        assert verbatim_normalize("hello   world") == "hello world"

    def test_preserves_case(self):
        assert verbatim_normalize("Hello World") == "Hello World"

    def test_folds_dashes(self):
        assert verbatim_normalize("2020–2023") == "2020-2023"  # en dash
        assert verbatim_normalize("self‐taught") == "self-taught"  # hyphen

    def test_folds_apostrophes(self):
        # U+2019 RIGHT SINGLE QUOTATION MARK → straight apostrophe
        assert verbatim_normalize("Master’s") == "Master's"

    def test_folds_quotes(self):
        # U+201C/U+201D LEFT/RIGHT DOUBLE QUOTATION MARK → straight double quote
        assert verbatim_normalize("“Hello”") == '"Hello"'

    def test_drops_soft_hyphens(self):
        assert verbatim_normalize("re­sume") == "resume"  # soft hyphen U+00AD

    def test_none_returns_empty(self):
        assert verbatim_normalize(None) == ""


# ---------------------------------------------------------------------------
# _record_ok
# ---------------------------------------------------------------------------


class TestRecordOk:

    def test_perfect_match(self):
        g = {"company": "Google", "position": "SWE"}
        p = {"company": "Google", "position": "SWE"}
        assert _record_ok(g, p, ["company", "position"]) is True

    def test_field_mismatch(self):
        g = {"company": "Google", "position": "SWE"}
        p = {"company": "Google", "position": "PM"}
        assert _record_ok(g, p, ["company", "position"]) is False

    def test_both_empty_fields_ok(self):
        g = {"company": "Google"}
        p = {"company": "Google"}
        assert _record_ok(g, p, ["company", "position"]) is True

    def test_one_empty_one_filled_fails(self):
        g = {"company": "Google", "position": "SWE"}
        p = {"company": "Google"}
        assert _record_ok(g, p, ["company", "position"]) is False

    def test_bullets_match(self):
        g = {"company": "X", "description": ["A", "B"]}
        p = {"company": "X", "description": ["A", "B"]}
        assert _record_ok(g, p, ["company"]) is True

    def test_bullets_count_mismatch(self):
        g = {"company": "X", "description": ["A", "B", "C"]}
        p = {"company": "X", "description": ["A", "B"]}
        assert _record_ok(g, p, ["company"]) is False

    def test_bullet_text_mismatch(self):
        g = {"company": "X", "description": ["A", "B"]}
        p = {"company": "X", "description": ["A", "C"]}
        assert _record_ok(g, p, ["company"]) is False

    def test_case_insensitive_default(self):
        g = {"company": "google"}
        p = {"company": "Google"}
        assert _record_ok(g, p, ["company"]) is True


# ---------------------------------------------------------------------------
# score_resume_strict — basics (singleton, empty-not-free)
# ---------------------------------------------------------------------------


class TestStrictBasics:

    def _sections(self):
        return (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email", "phone")),
        )

    def test_perfect(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "j@t.com", "phone": "123"}}
        pred = {"basics": {"fname": "John", "lname": "Doe", "email": "j@t.com", "phone": "123"}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0

    def test_empty_not_free(self):
        gt = {"basics": {"fname": "John", "lname": "Doe"}}
        pred = {"basics": {"fname": "John", "lname": "Doe"}}
        r = score_resume_strict(gt, pred, self._sections())
        # name matches (1 correct / 1 expected / 1 predicted), email+phone both empty → skipped
        assert r["sections"]["basics"]["correct"] == 1
        assert r["sections"]["basics"]["expected"] == 1
        assert r["f1"] == 1.0

    def test_name_mismatch(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "j@t.com"}}
        pred = {"basics": {"fname": "Jane", "lname": "Doe", "email": "j@t.com"}}
        r = score_resume_strict(gt, pred, self._sections())
        # name wrong, email correct → 1/2 correct
        assert r["sections"]["basics"]["correct"] == 1
        assert r["sections"]["basics"]["expected"] == 2


# ---------------------------------------------------------------------------
# score_resume_strict — entity list (record-level units)
# ---------------------------------------------------------------------------


class TestStrictEntityList:

    def _sections(self):
        return (
            SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position"), score_description=True),
        )

    def test_perfect_entity(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["experience"]["correct"] == 1
        assert r["f1"] == 1.0

    def test_partial_entity_fails_entirely(self):
        gt = {"experience": [{"company": "Google", "position": "SWE", "startYear": 2020}]}
        pred = {"experience": [{"company": "Google", "position": "SWE", "startYear": 2019}]}
        r = score_resume_strict(gt, pred, self._sections())
        # startYear mismatch → whole entity is wrong
        assert r["sections"]["experience"]["correct"] == 0

    def test_missed_entity(self):
        gt = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Meta", "position": "PM"},
        ]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["experience"]["expected"] == 2
        assert r["sections"]["experience"]["predicted"] == 1
        assert r["sections"]["experience"]["correct"] == 1

    def test_spurious_entity(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Fake", "position": "PM"},
        ]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["experience"]["expected"] == 1
        assert r["sections"]["experience"]["predicted"] == 2

    def test_empty_both_skipped(self):
        r = score_resume_strict({}, {}, self._sections())
        assert "experience" not in r["sections"]


# ---------------------------------------------------------------------------
# score_resume_strict — flat list (skills)
# ---------------------------------------------------------------------------


class TestStrictSkills:

    def _sections(self):
        return (SectionSpec("skills", SectionKind.FLAT_LIST, ()),)

    def test_perfect(self):
        gt = {"skills": [{"category": "Lang", "skills": ["Python", "Go"]}]}
        pred = {"skills": [{"category": "Lang", "skills": ["Python", "Go"]}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["skills"]["correct"] == 2
        assert r["f1"] == 1.0

    def test_missing_skill(self):
        gt = {"skills": [{"category": "X", "skills": ["Python", "Go", "Rust"]}]}
        pred = {"skills": [{"category": "X", "skills": ["Python", "Go"]}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["skills"]["correct"] == 2
        assert r["sections"]["skills"]["expected"] == 3


# ---------------------------------------------------------------------------
# score_resume_strict — personalSummary
# ---------------------------------------------------------------------------


class TestStrictPersonalSummary:

    def _sections(self):
        return (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)

    def test_perfect(self):
        gt = {"personalSummary": "Expert engineer with 10 years experience."}
        pred = {"personalSummary": "Expert engineer with 10 years experience."}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0

    def test_mismatch(self):
        gt = {"personalSummary": "Expert engineer."}
        pred = {"personalSummary": "Junior developer."}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["personalSummary"]["correct"] == 0

    def test_both_empty_skipped(self):
        r = score_resume_strict({"personalSummary": ""}, {"personalSummary": ""}, self._sections())
        assert "personalSummary" not in r["sections"]


# ---------------------------------------------------------------------------
# score_resume_strict — section averaging
# ---------------------------------------------------------------------------


class TestStrictSectionAveraging:

    def test_equal_weight(self):
        sections = (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email")),
            SectionSpec("skills", SectionKind.FLAT_LIST, ()),
        )
        gt = {
            "basics": {"fname": "John", "lname": "Doe", "email": "j@t.com"},
            "skills": [{"category": "X", "skills": ["Python", "Go"]}],
        }
        pred = {
            "basics": {"fname": "John", "lname": "Doe", "email": "j@t.com"},
            "skills": [{"category": "X", "skills": ["Python"]}],
        }
        r = score_resume_strict(gt, pred, sections)
        basics_f1 = r["sections"]["basics"]["f1"]
        skills_f1 = r["sections"]["skills"]["f1"]
        # Section-averaged: (basics_f1 + skills_f1) / 2
        expected_avg = (basics_f1 + skills_f1) / 2
        assert abs(r["f1"] - expected_avg) < 1e-6


# ---------------------------------------------------------------------------
# score_resume_strict — verbatim mode
# ---------------------------------------------------------------------------


class TestStrictVerbatim:

    def _sections(self):
        return (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email")),
        )

    def test_case_sensitive(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "J@T.com"}}
        pred = {"basics": {"fname": "john", "lname": "doe", "email": "j@t.com"}}
        # Non-verbatim: case insensitive → match
        r = score_resume_strict(gt, pred, self._sections(), verbatim=False)
        assert r["f1"] == 1.0

        # Verbatim: case sensitive → mismatch
        r = score_resume_strict(gt, pred, self._sections(), verbatim=True)
        assert r["f1"] < 1.0

    def test_typographic_folding(self):
        gt = {"basics": {"fname": "John", "lname": "Doe"}}
        pred = {"basics": {"fname": "John", "lname": "Doe"}}
        r = score_resume_strict(gt, pred, self._sections(), verbatim=True)
        assert r["f1"] == 1.0


# ---------------------------------------------------------------------------
# grade_single_strict integration
# ---------------------------------------------------------------------------


class TestGradeSingleStrict:

    def test_end_to_end(self):
        from resume_bench.grading.grader import grade_single_strict

        gt = {
            "basics": {"fname": "John", "lname": "Doe", "email": "j@test.com"},
            "experience": [{"company": "Google", "position": "SWE"}],
            "skills": [{"category": "Lang", "skills": ["Python"]}],
        }
        pred = {
            "basics": {"fname": "John", "lname": "Doe", "email": "j@test.com"},
            "experience": [{"company": "Google", "position": "SWE"}],
            "skills": [{"category": "Lang", "skills": ["Python"]}],
        }

        result = grade_single_strict(gt, pred)
        assert result["f1"] == 1.0
        assert "sections" in result

    def test_verbatim_flag(self):
        from resume_bench.grading.grader import grade_single_strict

        gt = {"basics": {"fname": "John", "lname": "Doe"}}
        pred = {"basics": {"fname": "john", "lname": "doe"}}

        r_default = grade_single_strict(gt, pred, verbatim=False)
        r_verbatim = grade_single_strict(gt, pred, verbatim=True)
        assert r_default["f1"] == 1.0
        assert r_verbatim["f1"] < 1.0

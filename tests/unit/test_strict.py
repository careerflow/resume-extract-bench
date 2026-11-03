"""Tests for ResumeBench-Strict scoring module."""

from resume_bench.grading.strict import (
    _direct_match,
    _field_passes,
    _identity_pass_rate,
    _text_rule,
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
        assert verbatim_normalize("Master’s") == "Master's"

    def test_folds_quotes(self):
        assert verbatim_normalize("“Hello”") == '"Hello"'

    def test_drops_soft_hyphens(self):
        assert verbatim_normalize("re­sume") == "resume"

    def test_none_returns_empty(self):
        assert verbatim_normalize(None) == ""


# ---------------------------------------------------------------------------
# _field_passes — type-appropriate field matching
# ---------------------------------------------------------------------------


class TestFieldPasses:

    def test_bool_match(self):
        assert _field_passes("inProgress", True, True) is True

    def test_bool_mismatch(self):
        assert _field_passes("inProgress", True, False) is False

    def test_int_match(self):
        assert _field_passes("startYear", 2020, 2020) is True

    def test_int_mismatch(self):
        assert _field_passes("startYear", 2020, 2021) is False

    def test_contact_case_insensitive(self):
        assert _field_passes("email", "John@Test.COM", "john@test.com") is True

    def test_location_case_insensitive(self):
        assert _field_passes("city", "New York", "new york") is True

    def test_entity_name_case_sensitive_verbatim(self):
        with _text_rule(True):
            assert _field_passes("company", "Google", "google") is False

    def test_entity_name_matches_exactly(self):
        with _text_rule(True):
            assert _field_passes("company", "Google", "Google") is True


# ---------------------------------------------------------------------------
# _identity_pass_rate
# ---------------------------------------------------------------------------


class TestIdentityPassRate:

    def test_all_match(self):
        g = {"company": "Google", "position": "SWE", "startYear": 2020, "endYear": 2023}
        p = {"company": "Google", "position": "SWE", "startYear": 2020, "endYear": 2023}
        with _text_rule(True):
            rate, total = _identity_pass_rate(g, p, "experience")
        assert rate == 1.0

    def test_one_mismatch(self):
        g = {"company": "Google", "position": "SWE", "startYear": 2020}
        p = {"company": "Meta", "position": "SWE", "startYear": 2020}
        with _text_rule(True):
            rate, total = _identity_pass_rate(g, p, "experience")
        assert rate < 1.0
        assert total == 3

    def test_both_empty_skipped(self):
        g = {"company": "Google"}
        p = {"company": "Google"}
        with _text_rule(True):
            rate, total = _identity_pass_rate(g, p, "experience")
        assert rate == 1.0
        assert total == 1


# ---------------------------------------------------------------------------
# _direct_match
# ---------------------------------------------------------------------------


class TestDirectMatch:

    def test_exact_match(self):
        gi = [{"company": "Google", "position": "SWE", "startYear": 2020}]
        pi = [{"company": "Google", "position": "SWE", "startYear": 2020}]
        with _text_rule(True):
            matched, missed, spurious = _direct_match(gi, pi, "experience", 1.0)
        assert matched == [(0, 0)]
        assert missed == []
        assert spurious == []

    def test_no_match_at_gate_1(self):
        gi = [{"company": "Google", "position": "SWE", "startYear": 2020}]
        pi = [{"company": "Meta", "position": "SWE", "startYear": 2020}]
        with _text_rule(True):
            matched, missed, spurious = _direct_match(gi, pi, "experience", 1.0)
        assert matched == []
        assert missed == [0]
        assert spurious == [0]

    def test_partial_match_at_lower_gate(self):
        gi = [{"company": "Google", "position": "SWE", "startYear": 2020}]
        pi = [{"company": "Meta", "position": "SWE", "startYear": 2020}]
        with _text_rule(True):
            matched, _, _ = _direct_match(gi, pi, "experience", 0.5)
        assert len(matched) == 1

    def test_multiple_entities(self):
        gi = [
            {"company": "Google", "position": "SWE", "startYear": 2020},
            {"company": "Meta", "position": "PM", "startYear": 2021},
        ]
        pi = [
            {"company": "Meta", "position": "PM", "startYear": 2021},
            {"company": "Google", "position": "SWE", "startYear": 2020},
        ]
        with _text_rule(True):
            matched, missed, spurious = _direct_match(gi, pi, "experience", 1.0)
        assert len(matched) == 2
        assert missed == []
        assert spurious == []


# ---------------------------------------------------------------------------
# score_resume_strict — basics (singleton, name composition)
# ---------------------------------------------------------------------------


class TestStrictBasics:

    def _sections(self):
        return (
            SectionSpec("basics", SectionKind.SINGLETON,
                        ("prefix", "fname", "lname", "suffix", "email", "phone")),
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
        assert r["sections"]["basics"]["correct"] == 1
        assert r["sections"]["basics"]["expected"] == 1
        assert r["f1"] == 1.0

    def test_name_mismatch(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "j@t.com"}}
        pred = {"basics": {"fname": "Jane", "lname": "Doe", "email": "j@t.com"}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["basics"]["correct"] == 1  # email matches
        assert r["sections"]["basics"]["expected"] == 2  # name + email

    def test_name_includes_prefix_suffix(self):
        gt = {"basics": {"prefix": "Dr.", "fname": "Ada", "lname": "Lovelace", "suffix": "PhD"}}
        pred = {"basics": {"prefix": "Dr.", "fname": "Ada", "lname": "Lovelace", "suffix": "PhD"}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0
        assert r["sections"]["basics"]["expected"] == 1  # only composed name

    def test_name_prefix_mismatch(self):
        gt = {"basics": {"prefix": "Dr.", "fname": "Ada", "lname": "Lovelace"}}
        pred = {"basics": {"fname": "Ada", "lname": "Lovelace"}}
        r = score_resume_strict(gt, pred, self._sections())
        # "Dr. Ada Lovelace" vs "Ada Lovelace" → mismatch
        assert r["sections"]["basics"]["correct"] == 0

    def test_contact_case_insensitive(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "J@T.COM"}}
        pred = {"basics": {"fname": "John", "lname": "Doe", "email": "j@t.com"}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["basics"]["correct"] == 2  # name + email


# ---------------------------------------------------------------------------
# score_resume_strict — entity list (per-field scoring)
# ---------------------------------------------------------------------------


class TestStrictEntityList:

    def _sections(self):
        return (
            SectionSpec("experience", SectionKind.ENTITY_LIST,
                        ("company", "position"), score_description=True),
        )

    def test_perfect_entity(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0

    def test_per_field_partial_credit(self):
        gt = {
            "experience": [{"company": "Google", "position": "SWE", "startYear": 2020,
                            "description": ["A", "B", "C"]}],
        }
        pred = {
            "experience": [{"company": "Google", "position": "SWE", "startYear": 2020,
                            "description": ["A", "WRONG", "C"]}],
        }
        r = score_resume_strict(gt, pred, self._sections())
        sec = r["sections"]["experience"]
        assert sec["correct"] == 5  # company + position + startYear + A + C
        assert sec["expected"] == 6

    def test_identity_mismatch_gives_zero(self):
        gt = {"experience": [{"company": "Google", "position": "SWE", "startYear": 2020}]}
        pred = {"experience": [{"company": "Meta", "position": "PM", "startYear": 2021}]}
        r = score_resume_strict(gt, pred, self._sections(), gate=1.0)
        assert r["sections"]["experience"]["correct"] == 0

    def test_missed_entity(self):
        gt = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Meta", "position": "PM"},
        ]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        r = score_resume_strict(gt, pred, self._sections())
        sec = r["sections"]["experience"]
        assert sec["expected"] > sec["predicted"]

    def test_spurious_entity(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Fake", "position": "PM"},
        ]}
        r = score_resume_strict(gt, pred, self._sections())
        sec = r["sections"]["experience"]
        assert sec["predicted"] > sec["expected"]

    def test_empty_both_skipped(self):
        r = score_resume_strict({}, {}, self._sections())
        assert "experience" not in r["sections"]

    def test_experience_order_independent(self):
        gt = {
            "experience": [
                {"company": "Google", "position": "SWE"},
                {"company": "Meta", "position": "PM"},
            ]
        }
        pred = {
            "experience": [
                {"company": "Meta", "position": "PM"},
                {"company": "Google", "position": "SWE"},
            ]
        }
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0


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

    def test_skills_order_independent(self):
        gt = {"skills": [{"skills": ["Python", "Go"]}]}
        pred = {"skills": [{"skills": ["Go", "Python"]}]}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0


# ---------------------------------------------------------------------------
# score_resume_strict — keyword sections (Hungarian alignment)
# ---------------------------------------------------------------------------


class TestStrictKeywordSections:

    def test_languages_order_independent(self):
        gt = {"languages": [{"name": "French"}, {"name": "English"}]}
        pred = {"languages": [{"name": "English"}, {"name": "French"}]}
        sections = (SectionSpec("languages", SectionKind.ENTITY_LIST, ("name",)),)
        r = score_resume_strict(gt, pred, sections)
        assert r["sections"]["languages"]["correct"] == 2

    def test_total_mismatch_rejected(self):
        """Completely mismatched entities should not be paired by Hungarian."""
        gt = {"languages": [{"name": "French"}]}
        pred = {"languages": [{"name": "Mandarin"}]}
        sections = (SectionSpec("languages", SectionKind.ENTITY_LIST, ("name",)),)
        r = score_resume_strict(gt, pred, sections)
        sec = r["sections"]["languages"]
        assert sec["correct"] == 0
        assert sec["expected"] == 1
        assert sec["predicted"] == 1

    def test_partial_match_still_paired(self):
        """Entities with at least one matching field should still be paired."""
        gt = {"publications": [{"name": "Paper A", "date": "2023"}]}
        pred = {"publications": [{"name": "Paper A", "date": "2024"}]}
        sections = (SectionSpec("publications", SectionKind.ENTITY_LIST, ("name",)),)
        r = score_resume_strict(gt, pred, sections)
        sec = r["sections"]["publications"]
        assert sec["correct"] == 1
        assert sec["expected"] == 2


# ---------------------------------------------------------------------------
# score_resume_strict — personalSummary
# ---------------------------------------------------------------------------


class TestStrictPersonalSummary:

    def _sections(self):
        return (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)

    def test_perfect(self):
        gt = {"personalSummary": {"text": "Expert engineer with 10 years experience."}}
        pred = {"personalSummary": {"text": "Expert engineer with 10 years experience."}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["f1"] == 1.0

    def test_mismatch(self):
        gt = {"personalSummary": {"text": "Expert engineer."}}
        pred = {"personalSummary": {"text": "Junior developer."}}
        r = score_resume_strict(gt, pred, self._sections())
        assert r["sections"]["personalSummary"]["correct"] == 0

    def test_both_empty_bare_string_skipped(self):
        r = score_resume_strict(
            {"personalSummary": ""},
            {"personalSummary": ""},
            self._sections(),
        )
        assert "personalSummary" not in r["sections"]


# ---------------------------------------------------------------------------
# score_resume_strict — section averaging
# ---------------------------------------------------------------------------


class TestStrictSectionAveraging:

    def test_equal_weight(self):
        sections = (
            SectionSpec("basics", SectionKind.SINGLETON,
                        ("fname", "lname", "email")),
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
        expected_avg = (basics_f1 + skills_f1) / 2
        assert abs(r["f1"] - expected_avg) < 1e-6


# ---------------------------------------------------------------------------
# score_resume_strict — verbatim mode
# ---------------------------------------------------------------------------


class TestStrictVerbatim:

    def _sections(self):
        return (
            SectionSpec("basics", SectionKind.SINGLETON,
                        ("fname", "lname", "email")),
        )

    def test_case_sensitive(self):
        gt = {"basics": {"fname": "John", "lname": "Doe"}}
        pred = {"basics": {"fname": "john", "lname": "doe"}}
        r_nonverbatim = score_resume_strict(gt, pred, self._sections(), verbatim=False)
        r_verbatim = score_resume_strict(gt, pred, self._sections(), verbatim=True)
        assert r_nonverbatim["f1"] == 1.0
        assert r_verbatim["f1"] < 1.0


# ---------------------------------------------------------------------------
# Thread-safety: concurrent scoring calls with different verbatim settings
# ---------------------------------------------------------------------------


class TestStrictConcurrency:

    def test_concurrent_verbatim_modes(self):
        """Concurrent scoring calls with different verbatim settings must not interfere."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sections = (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname")),
        )
        gt = {"basics": {"fname": "John", "lname": "Doe"}}
        pred_lower = {"basics": {"fname": "john", "lname": "doe"}}

        results = {"verbatim": [], "nonverbatim": []}

        def run_verbatim():
            r = score_resume_strict(gt, pred_lower, sections, verbatim=True)
            return "verbatim", r["f1"]

        def run_nonverbatim():
            r = score_resume_strict(gt, pred_lower, sections, verbatim=False)
            return "nonverbatim", r["f1"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = []
            for _ in range(20):
                futs.append(pool.submit(run_verbatim))
                futs.append(pool.submit(run_nonverbatim))
            for fut in as_completed(futs):
                mode, f1 = fut.result()
                results[mode].append(f1)

        assert all(f1 < 1.0 for f1 in results["verbatim"]), "verbatim should NOT match case-different names"
        assert all(f1 == 1.0 for f1 in results["nonverbatim"]), "nonverbatim should match case-different names"


# ---------------------------------------------------------------------------
# score_resume_strict — gate threshold
# ---------------------------------------------------------------------------


class TestStrictGate:

    def _sections(self):
        return (
            SectionSpec("experience", SectionKind.ENTITY_LIST,
                        ("company", "position"), score_description=True),
        )

    def test_gate_zero_matches_everything(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [{"company": "Wrong", "position": "Wrong"}]}
        r = score_resume_strict(gt, pred, self._sections(), gate=0.0)
        assert r["sections"]["experience"]["correct"] == 0
        assert r["sections"]["experience"]["expected"] == 2
        assert r["sections"]["experience"]["predicted"] == 2


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
        result = grade_single_strict(gt, gt)
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

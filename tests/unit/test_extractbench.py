"""Tests for the ExtractBench-exact scoring module."""

from resume_bench.grading.extractbench import (
    _eb_align_entities,
    _eb_cell_mismatch_count,
    _eb_count_entity_cells,
    _eb_normalize,
    _eb_str_match,
    eb_score_resume,
)
from resume_bench.grading.models import CellCounts, ExtractBenchScore
from resume_bench.schema.sections import SectionKind, SectionSpec


# ---------------------------------------------------------------------------
# _eb_normalize
# ---------------------------------------------------------------------------


class TestEbNormalize:

    def test_whitespace_collapsing(self):
        assert _eb_normalize("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert _eb_normalize("hello\t\nworld") == "hello world"

    def test_case_insensitive(self):
        assert _eb_normalize("Hello World") == "hello world"

    def test_strips_outer_whitespace(self):
        assert _eb_normalize("  hello  ") == "hello"

    def test_none_returns_empty(self):
        assert _eb_normalize(None) == ""

    def test_integer_coerced(self):
        assert _eb_normalize(42) == "42"


# ---------------------------------------------------------------------------
# _eb_str_match
# ---------------------------------------------------------------------------


class TestEbStrMatch:

    def test_exact_match(self):
        assert _eb_str_match("hello", "hello") == 1.0

    def test_case_insensitive_match(self):
        assert _eb_str_match("Hello", "hello") == 1.0

    def test_whitespace_normalized_match(self):
        assert _eb_str_match("hello   world", "hello world") == 1.0

    def test_near_miss(self):
        assert _eb_str_match("hello", "helo") == 0.0

    def test_none_vs_empty(self):
        assert _eb_str_match(None, "") == 1.0

    def test_none_vs_value(self):
        assert _eb_str_match(None, "hello") == 0.0


# ---------------------------------------------------------------------------
# _eb_cell_mismatch_count
# ---------------------------------------------------------------------------


class TestEbCellMismatchCount:

    def test_perfect_match_scalars(self):
        gt = {"company": "Google", "position": "SWE"}
        pred = {"company": "Google", "position": "SWE"}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["company", "position"])
        assert mismatches == 0
        assert total == 2

    def test_one_mismatch(self):
        gt = {"company": "Google", "position": "SWE"}
        pred = {"company": "Google", "position": "Engineer"}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["company", "position"])
        assert mismatches == 1
        assert total == 2

    def test_both_null_is_match(self):
        gt = {}
        pred = {}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["company"])
        assert mismatches == 0
        assert total == 1

    def test_gt_null_pred_has_value(self):
        gt = {}
        pred = {"company": "Google"}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["company"])
        assert mismatches == 1
        assert total == 1

    def test_boolean_match(self):
        gt = {"inProgress": True}
        pred = {"inProgress": True}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["inProgress"])
        assert mismatches == 0

    def test_boolean_mismatch(self):
        gt = {"inProgress": True}
        pred = {"inProgress": False}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["inProgress"])
        assert mismatches == 1

    def test_integer_match(self):
        gt = {"startYear": 2020}
        pred = {"startYear": 2020}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["startYear"])
        assert mismatches == 0

    def test_integer_mismatch(self):
        gt = {"startYear": 2020}
        pred = {"startYear": 2019}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, ["startYear"])
        assert mismatches == 1

    def test_description_bullets(self):
        gt = {"description": ["Built search", "Led team"]}
        pred = {"description": ["Built search", "Led team"]}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, [])
        assert mismatches == 0
        assert total == 2

    def test_description_missing_bullet(self):
        gt = {"description": ["A", "B", "C"]}
        pred = {"description": ["A", "B"]}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, [])
        assert mismatches == 1  # "C" vs ""
        assert total == 3

    def test_description_extra_bullet(self):
        gt = {"description": ["A"]}
        pred = {"description": ["A", "Extra"]}
        mismatches, total = _eb_cell_mismatch_count(gt, pred, [])
        assert mismatches == 1  # "" vs "Extra"
        assert total == 2


# ---------------------------------------------------------------------------
# _eb_align_entities
# ---------------------------------------------------------------------------


class TestEbAlignEntities:

    def test_perfect_match(self):
        gt = [{"company": "Google", "position": "SWE"}]
        pred = [{"company": "Google", "position": "SWE"}]
        matched, missed, spurious = _eb_align_entities(gt, pred, ["company", "position"])
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert missed == []
        assert spurious == []

    def test_empty_gt(self):
        pred = [{"company": "Google"}]
        matched, missed, spurious = _eb_align_entities([], pred, ["company"])
        assert matched == []
        assert missed == []
        assert spurious == [0]

    def test_empty_pred(self):
        gt = [{"company": "Google"}]
        matched, missed, spurious = _eb_align_entities(gt, [], ["company"])
        assert matched == []
        assert missed == [0]
        assert spurious == []

    def test_both_empty(self):
        matched, missed, spurious = _eb_align_entities([], [], ["company"])
        assert matched == []
        assert missed == []
        assert spurious == []

    def test_partial_match_no_threshold(self):
        """Unlike fuzzy alignment, EB keeps all assignments (no threshold)."""
        gt = [{"company": "Google"}]
        pred = [{"company": "Amazon"}]
        matched, missed, spurious = _eb_align_entities(gt, pred, ["company"])
        # Even though company doesn't match, the assignment is still kept
        assert len(matched) == 1
        assert missed == []
        assert spurious == []

    def test_uneven_counts(self):
        gt = [{"company": "Google"}, {"company": "Meta"}]
        pred = [{"company": "Google"}]
        matched, missed, spurious = _eb_align_entities(gt, pred, ["company"])
        assert len(matched) == 1
        assert len(missed) == 1
        assert spurious == []


# ---------------------------------------------------------------------------
# _eb_count_entity_cells
# ---------------------------------------------------------------------------


class TestEbCountEntityCells:

    def test_scalars_only(self):
        entity = {"company": "Google", "position": "SWE"}
        count = _eb_count_entity_cells(entity, ["company", "position"])
        assert count == 2

    def test_with_description(self):
        entity = {"company": "Google", "description": ["A", "B", "C"]}
        count = _eb_count_entity_cells(entity, ["company"])
        # 1 scalar + 3 bullets = 4
        assert count == 4

    def test_empty_entity_minimum_one(self):
        count = _eb_count_entity_cells({}, [])
        assert count == 1


# ---------------------------------------------------------------------------
# eb_score_resume
# ---------------------------------------------------------------------------


class TestEbScoreResume:

    def _make_sections(self):
        """Build a small set of sections for testing."""
        return (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email")),
            SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position"), score_description=True),
            SectionSpec("skills", SectionKind.FLAT_LIST, ()),
            SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),
        )

    def test_empty_resume(self):
        cells = eb_score_resume({}, {}, self._make_sections())
        # Singleton (basics) still scores key_fields even when both sides
        # are empty — each null/null pair counts as correct.  Entity-list
        # and flat-list sections with empty GT+pred are skipped entirely.
        # basics has fname+lname (joined → 1 name cell) + email → 2 cells
        assert cells.correct == 2
        assert cells.expected == 2
        assert cells.predicted == 2
        assert cells.f1 == 1.0

    def test_singleton_perfect(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "john@test.com"}}
        pred = {"basics": {"fname": "John", "lname": "Doe", "email": "john@test.com"}}
        sections = (SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email")),)
        cells = eb_score_resume(gt, pred, sections)
        # name (joined) + email = 2 cells, all correct
        assert cells.correct == 2
        assert cells.expected == 2
        assert cells.predicted == 2
        assert cells.f1 == 1.0

    def test_singleton_name_mismatch(self):
        gt = {"basics": {"fname": "John", "lname": "Doe", "email": "john@test.com"}}
        pred = {"basics": {"fname": "Jane", "lname": "Doe", "email": "john@test.com"}}
        sections = (SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname", "email")),)
        cells = eb_score_resume(gt, pred, sections)
        # name mismatch (John Doe vs Jane Doe), email correct
        assert cells.correct == 1
        assert cells.expected == 2
        assert cells.predicted == 2

    def test_entity_list_perfect(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        sections = (SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position")),)
        cells = eb_score_resume(gt, pred, sections)
        # SECTION_SCHEMA_FIELDS["experience"] has 10 fields; both null = correct
        assert cells.correct == cells.expected == cells.predicted
        assert cells.f1 == 1.0

    def test_entity_list_with_description(self):
        gt = {"experience": [{"company": "Google", "description": ["Built search", "Led team"]}]}
        pred = {"experience": [{"company": "Google", "description": ["Built search", "Led team"]}]}
        sections = (SectionSpec("experience", SectionKind.ENTITY_LIST, ("company",), score_description=True),)
        cells = eb_score_resume(gt, pred, sections)
        # SECTION_SCHEMA_FIELDS["experience"] has multiple fields; bullets are separate cells.
        # company is one of the s_fields, and description bullets are 2 cells.
        assert cells.f1 == 1.0

    def test_entity_list_missed_entity(self):
        gt = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Meta", "position": "PM"},
        ]}
        pred = {"experience": [{"company": "Google", "position": "SWE"}]}
        sections = (SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position")),)
        cells = eb_score_resume(gt, pred, sections)
        # Matched: Google (all fields correct)
        # Missed: Meta (cells go to expected only)
        assert cells.expected > cells.predicted
        assert cells.f1 < 1.0

    def test_entity_list_spurious_entity(self):
        gt = {"experience": [{"company": "Google", "position": "SWE"}]}
        pred = {"experience": [
            {"company": "Google", "position": "SWE"},
            {"company": "Fake", "position": "PM"},
        ]}
        sections = (SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position")),)
        cells = eb_score_resume(gt, pred, sections)
        # Matched: Google
        # Spurious: Fake (cells go to predicted only)
        assert cells.predicted > cells.expected
        assert cells.f1 < 1.0

    def test_flat_list_perfect(self):
        gt = {"skills": [{"category": "Lang", "skills": ["Python", "Go"]}]}
        pred = {"skills": [{"category": "Lang", "skills": ["Python", "Go"]}]}
        sections = (SectionSpec("skills", SectionKind.FLAT_LIST, ()),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 2
        assert cells.expected == 2
        assert cells.predicted == 2
        assert cells.f1 == 1.0

    def test_flat_list_missing_skill(self):
        gt = {"skills": [{"category": "X", "skills": ["Python", "Go", "Rust"]}]}
        pred = {"skills": [{"category": "X", "skills": ["Python", "Go"]}]}
        sections = (SectionSpec("skills", SectionKind.FLAT_LIST, ()),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 2
        assert cells.expected == 3
        assert cells.predicted == 2

    def test_flat_list_hallucinated_skill(self):
        gt = {"skills": [{"category": "X", "skills": ["Python"]}]}
        pred = {"skills": [{"category": "X", "skills": ["Python", "FakeSkill"]}]}
        sections = (SectionSpec("skills", SectionKind.FLAT_LIST, ()),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 1
        assert cells.expected == 1
        assert cells.predicted == 2

    def test_personal_summary_perfect(self):
        gt = {"personalSummary": "Experienced engineer with 10 years in backend systems."}
        pred = {"personalSummary": "Experienced engineer with 10 years in backend systems."}
        sections = (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 1
        assert cells.f1 == 1.0

    def test_personal_summary_mismatch(self):
        gt = {"personalSummary": "Experienced engineer with 10 years in backend systems."}
        pred = {"personalSummary": "Junior developer looking for opportunities."}
        sections = (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 0
        assert cells.expected == 1
        assert cells.predicted == 1

    def test_personal_summary_gt_only(self):
        """GT has summary, pred is empty — cells go to expected only (like entity-list miss)."""
        gt = {"personalSummary": "Experienced engineer."}
        pred = {"personalSummary": ""}
        sections = (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 0
        assert cells.expected == 1
        assert cells.predicted == 0  # NOT 1 — matches internal repo behavior

    def test_personal_summary_pred_only(self):
        """Pred has summary, GT is empty — cells go to predicted only (like entity-list spurious)."""
        gt = {"personalSummary": ""}
        pred = {"personalSummary": "Hallucinated summary."}
        sections = (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)
        cells = eb_score_resume(gt, pred, sections)
        assert cells.correct == 0
        assert cells.expected == 0  # NOT 1 — matches internal repo behavior
        assert cells.predicted == 1

    def test_personal_summary_both_empty(self):
        gt = {"personalSummary": ""}
        pred = {"personalSummary": ""}
        sections = (SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),)
        cells = eb_score_resume(gt, pred, sections)
        # Both empty → skipped
        assert cells.correct == 0
        assert cells.expected == 0

    def test_exclude_sections(self):
        gt = {
            "basics": {"fname": "John", "lname": "Doe"},
            "experience": [{"company": "Google", "position": "SWE"}],
        }
        pred = {
            "basics": {"fname": "John", "lname": "Doe"},
            "experience": [{"company": "Wrong", "position": "Wrong"}],
        }
        sections = (
            SectionSpec("basics", SectionKind.SINGLETON, ("fname", "lname")),
            SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position")),
        )
        cells = eb_score_resume(gt, pred, sections, exclude_sections={"experience"})
        # Only basics scored — name (joined) = 1 cell, perfect
        assert cells.correct == 1
        assert cells.expected == 1
        assert cells.f1 == 1.0

    def test_description_bullets_in_entity_scoring(self):
        """Verify description bullets are counted as individual cells."""
        gt = {"experience": [{"company": "Google", "description": ["A", "B", "C"]}]}
        pred = {"experience": [{"company": "Google", "description": ["A", "B"]}]}
        sections = (SectionSpec("experience", SectionKind.ENTITY_LIST, ("company",)),)
        cells = eb_score_resume(gt, pred, sections)
        # Schema fields for experience (from SECTION_SCHEMA_FIELDS) + 3 GT bullets / 2 pred bullets
        # The missing bullet "C" adds 1 to expected but not predicted
        assert cells.expected > cells.predicted
        assert cells.recall < 1.0


# ---------------------------------------------------------------------------
# CellCounts
# ---------------------------------------------------------------------------


class TestCellCounts:

    def test_perfect_scores(self):
        c = CellCounts(correct=10, expected=10, predicted=10)
        assert c.precision == 1.0
        assert c.recall == 1.0
        assert c.f1 == 1.0

    def test_no_predictions(self):
        c = CellCounts(correct=0, expected=10, predicted=0)
        assert c.precision == 0.0
        assert c.recall == 0.0
        assert c.f1 == 0.0

    def test_partial(self):
        c = CellCounts(correct=5, expected=10, predicted=8)
        assert c.precision == 5 / 8
        assert c.recall == 5 / 10
        p, r = 5 / 8, 5 / 10
        assert abs(c.f1 - 2 * p * r / (p + r)) < 1e-9


# ---------------------------------------------------------------------------
# grade_single_extractbench integration
# ---------------------------------------------------------------------------


class TestGradeSingleExtractBench:

    def test_end_to_end(self):
        from resume_bench.grading.grader import grade_single_extractbench

        gt = {
            "basics": {"fname": "John", "lname": "Doe", "email": "john@test.com"},
            "experience": [{"company": "Google", "position": "SWE"}],
            "skills": [{"category": "Lang", "skills": ["Python"]}],
        }
        pred = {
            "basics": {"fname": "John", "lname": "Doe", "email": "john@test.com"},
            "experience": [{"company": "Google", "position": "SWE"}],
            "skills": [{"category": "Lang", "skills": ["Python"]}],
        }

        result = grade_single_extractbench(gt, pred)
        assert isinstance(result, ExtractBenchScore)
        assert result.completed is True
        assert result.cells.correct > 0
        # Perfect match → F1 = 1.0
        assert result.cells.f1 == 1.0

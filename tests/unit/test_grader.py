from resume_bench.grading.grader import grade_single


class TestGradeSingle:

    def test_full_resume(self):
        gt = {
            "basics": {"fname": "Jane", "lname": "Smith", "email": "jane@test.com"},
            "experience": [
                {"company": "Google", "position": "SWE", "description": ["Built features"]},
            ],
            "education": [
                {"institution": "MIT", "area": "CS"},
            ],
            "skills": [{"category": "Lang", "skills": ["Python", "Go"]}],
            "personalSummary": "Experienced engineer.",
        }
        pred = {
            "basics": {"fname": "Jane", "lname": "Smith", "email": "jane@test.com"},
            "experience": [
                {"company": "Google", "position": "SWE", "description": ["Built features"]},
            ],
            "education": [
                {"institution": "MIT", "area": "CS"},
            ],
            "skills": [{"category": "Lang", "skills": ["Python", "Go"]}],
            "personalSummary": "Experienced engineer.",
        }

        score = grade_single(gt, pred)

        assert score.macro_entity_f1 > 0.9

    def test_empty_resume(self):
        score = grade_single({}, {})

        assert score.macro_entity_f1 == 0.0
        for sec in score.sections.values():
            assert sec.is_vacuous is True

    def test_missing_sections(self):
        gt = {
            "basics": {"fname": "John"},
            "experience": [{"company": "Google", "position": "SWE"}],
        }
        pred = {
            "basics": {"fname": "John"},
        }

        score = grade_single(gt, pred)

        assert "basics" in score.sections
        assert "experience" in score.sections
        assert score.sections["experience"].omission_rate == 1.0

    def test_personal_summary_scoring(self):
        gt = {"personalSummary": "Experienced software engineer with 10 years."}
        pred = {"personalSummary": "Experienced software engineer with 10 years of experience."}

        score = grade_single(gt, pred)

        assert "personalSummary" in score.sections
        assert score.sections["personalSummary"].f1 > 0.7

    def test_non_vacuous_sections(self):
        gt = {
            "basics": {"fname": "John"},
            "experience": [],
            "education": [],
        }
        pred = {
            "basics": {"fname": "John"},
            "experience": [],
            "education": [],
        }

        score = grade_single(gt, pred)

        nv = score.non_vacuous_sections
        assert "experience" not in nv
        assert "education" not in nv

    def test_basics_included_in_headline(self):
        gt = {
            "basics": {"fname": "Jane", "lname": "Smith"},
            "experience": [{"company": "Google", "position": "SWE"}],
        }
        pred = {
            "basics": {"fname": "Jane", "lname": "Smith"},
            "experience": [{"company": "Google", "position": "SWE"}],
        }

        score = grade_single(gt, pred)

        assert score.sections["basics"].in_headline is True
        assert "basics" in score.headline_sections
        assert score.basics_field_accuracy > 0.9

    def test_wrapped_sections(self):
        """Grader handles wrapped {sectionTitle, items} format."""
        gt = {
            "basics": {"fname": "Jane", "lname": "Smith", "email": "jane@test.com"},
            "experience": {
                "sectionTitle": "Experience",
                "items": [
                    {"company": "Google", "position": "SWE", "description": ["Built features"]},
                ],
            },
            "education": {
                "sectionTitle": "Education",
                "items": [
                    {"institution": "MIT", "area": "CS"},
                ],
            },
            "skills": {
                "sectionTitle": "Skills",
                "items": [{"category": "Lang", "skills": ["Python", "Go"]}],
            },
            "personalSummary": {
                "sectionTitle": "Summary",
                "text": "Experienced engineer.",
            },
        }
        pred = {
            "basics": {"fname": "Jane", "lname": "Smith", "email": "jane@test.com"},
            "experience": {
                "sectionTitle": "Experience",
                "items": [
                    {"company": "Google", "position": "SWE", "description": ["Built features"]},
                ],
            },
            "education": {
                "sectionTitle": "Education",
                "items": [
                    {"institution": "MIT", "area": "CS"},
                ],
            },
            "skills": {
                "sectionTitle": "Skills",
                "items": [{"category": "Lang", "skills": ["Python", "Go"]}],
            },
            "personalSummary": {
                "sectionTitle": "Summary",
                "text": "Experienced engineer.",
            },
        }

        score = grade_single(gt, pred)

        assert score.macro_entity_f1 > 0.9

    def test_wrapped_personal_summary(self):
        """personalSummary accepts wrapped {sectionTitle, text} format."""
        gt = {"personalSummary": {"sectionTitle": "Profile", "text": "Software engineer."}}
        pred = {"personalSummary": {"sectionTitle": "Profile", "text": "Software engineer."}}

        score = grade_single(gt, pred)

        assert score.sections["personalSummary"].f1 == 1.0

    def test_mixed_wrapped_and_flat(self):
        """Grader handles GT in wrapped format and prediction in flat format."""
        gt = {
            "experience": {
                "sectionTitle": "Work History",
                "items": [
                    {"company": "Google", "position": "SWE"},
                ],
            },
        }
        pred = {
            "experience": [
                {"company": "Google", "position": "SWE"},
            ],
        }

        score = grade_single(gt, pred)

        assert score.sections["experience"].f1 > 0.9

    def test_failed_resume_scores_zero(self):
        from resume_bench.grading.models import ResumeScore, SectionScore
        from resume_bench.schema.sections import SECTIONS

        failed = ResumeScore(resume_id="failed_001", completed=False)

        for spec in SECTIONS:
            failed.sections[spec.name] = SectionScore()

        assert failed.macro_entity_f1 == 0.0
        assert failed.completed is False

from resume_bench.grading.metrics import (
    _entity_quality,
    _positional_bullet_score,
    score_entity_list,
    score_flat_list,
    score_singleton,
)


class TestScoreSingleton:

    def test_perfect_match(self):
        gt = {"fname": "John", "lname": "Doe"}
        pred = {"fname": "John", "lname": "Doe"}

        score = score_singleton(gt, pred, ("fname", "lname"))

        assert score.f1 == 1.0

    def test_partial_match(self):
        gt = {"fname": "John", "lname": "Doe", "email": "john@test.com", "phone": "555-1234"}
        pred = {"fname": "John", "lname": "Doe", "email": "", "phone": "555-1234"}

        score = score_singleton(gt, pred, ("fname", "lname", "email", "phone"))

        # 3 fields after join: name=1.0, email=0.0, phone=1.0 → avg ≈ 0.667
        assert 0.5 < score.f1 < 1.0

    def test_both_empty_is_vacuous(self):
        gt = {}
        pred = {}

        score = score_singleton(gt, pred, ("fname", "lname"))

        assert score.is_vacuous is True
        assert score.f1 == 1.0

    def test_boolean_fields(self):
        gt = {"hasPersonalPhoto": True}
        pred = {"hasPersonalPhoto": True}

        score = score_singleton(gt, pred, ("hasPersonalPhoto",))

        assert score.f1 == 1.0

    def test_boolean_mismatch(self):
        gt = {"hasPersonalPhoto": True}
        pred = {"hasPersonalPhoto": False}

        score = score_singleton(gt, pred, ("hasPersonalPhoto",))

        assert score.f1 == 0.0

    def test_field_accuracy_reported(self):
        gt = {"fname": "John", "lname": "Doe"}
        pred = {"fname": "John", "lname": "Smith"}

        score = score_singleton(gt, pred, ("fname", "lname"))

        # fname + lname are joined into a single "name" field
        assert "name" in score.field_accuracy
        assert "fname" not in score.field_accuracy
        # "John Doe" vs "John Smith" — partial match
        assert score.field_accuracy["name"] < 1.0

    def test_name_join_different_split(self):
        """Models that split the name differently should not be penalised."""
        gt = {"fname": "Jean Marie", "lname": "Schiraldi"}
        pred = {"fname": "Jean", "lname": "Marie Schiraldi"}

        score = score_singleton(gt, pred, ("fname", "lname"))

        # Both produce "Jean Marie Schiraldi" after joining
        assert score.field_accuracy["name"] == 1.0
        assert score.f1 == 1.0


class TestScoreFlatList:

    def test_perfect_match(self):
        gt = [{"category": "Lang", "skills": ["Python", "Go"]}]
        pred = [{"category": "Lang", "skills": ["Python", "Go"]}]

        score = score_flat_list(gt, pred)

        assert score.f1 == 1.0

    def test_both_empty_is_vacuous(self):
        score = score_flat_list([], [])

        assert score.is_vacuous is True
        assert score.f1 == 1.0

    def test_gt_empty_all_hallucinated(self):
        pred = [{"category": "X", "skills": ["Python"]}]

        score = score_flat_list([], pred)

        assert score.hallucination_rate == 1.0
        assert score.f1 == 0.0

    def test_pred_empty_all_omitted(self):
        gt = [{"category": "X", "skills": ["Python"]}]

        score = score_flat_list(gt, [])

        assert score.omission_rate == 1.0
        assert score.f1 == 0.0

    def test_deduplication(self):
        gt = [
            {"category": "A", "skills": ["Python"]},
            {"category": "B", "skills": ["Python"]},
        ]
        pred = [{"category": "C", "skills": ["Python"]}]

        score = score_flat_list(gt, pred)

        assert score.f1 == 1.0

    def test_case_insensitive(self):
        gt = [{"category": "X", "skills": ["Python"]}]
        pred = [{"category": "X", "skills": ["python"]}]

        score = score_flat_list(gt, pred)

        assert score.f1 == 1.0


class TestScoreEntityList:

    def test_perfect_match(self):
        gt = [{"company": "Google", "position": "SWE"}]
        pred = [{"company": "Google", "position": "SWE"}]

        score = score_entity_list(gt, pred, ("company", "position"))

        assert score.f1 == 1.0

    def test_both_empty_is_vacuous(self):
        score = score_entity_list([], [], ("company",))

        assert score.is_vacuous is True

    def test_gt_empty_all_hallucinated(self):
        pred = [{"company": "Google", "position": "SWE"}]

        score = score_entity_list([], pred, ("company",))

        assert score.hallucination_rate == 1.0

    def test_pred_empty_all_omitted(self):
        gt = [{"company": "Google", "position": "SWE"}]

        score = score_entity_list(gt, [], ("company",))

        assert score.omission_rate == 1.0

    def test_description_scoring(self):
        gt = [{"company": "Google", "description": ["Built search features", "Led team"]}]
        pred = [{"company": "Google", "description": ["Built search features", "Led team"]}]

        score = score_entity_list(
            gt, pred, ("company",), score_description=True,
        )

        assert score.description_token_f1 is not None
        assert score.description_token_f1 == 1.0

    def test_description_reordered_scores_lower(self):
        gt = [{"company": "Google", "description": ["Built search features", "Led team of 5"]}]
        pred = [{"company": "Google", "description": ["Led team of 5", "Built search features"]}]

        score = score_entity_list(
            gt, pred, ("company",), score_description=True,
        )

        assert score.description_token_f1 is not None
        assert score.description_token_f1 < 1.0

    def test_description_missing_bullet_penalized(self):
        gt = [{"company": "Google", "description": ["A", "B", "C"]}]
        pred = [{"company": "Google", "description": ["A", "B"]}]

        score = score_entity_list(
            gt, pred, ("company",), score_description=True,
        )

        assert score.description_token_f1 is not None
        # (1.0 + 1.0 + 0.0) / 3 ≈ 0.667
        assert abs(score.description_token_f1 - 2 / 3) < 0.01

    def test_description_extra_bullet_penalized(self):
        gt = [{"company": "Google", "description": ["A"]}]
        pred = [{"company": "Google", "description": ["A", "Hallucinated"]}]

        score = score_entity_list(
            gt, pred, ("company",), score_description=True,
        )

        assert score.description_token_f1 is not None
        assert score.description_token_f1 < 1.0

    def test_no_description_scoring_by_default(self):
        gt = [{"company": "Google", "description": ["text"]}]
        pred = [{"company": "Google", "description": ["text"]}]

        score = score_entity_list(gt, pred, ("company",))

        assert score.description_token_f1 is None

    def test_field_accuracy(self):
        gt = [{"company": "Google", "position": "SWE"}]
        pred = [{"company": "Google", "position": "Engineer"}]

        score = score_entity_list(gt, pred, ("company", "position"))

        assert score.field_accuracy["company"] == 1.0
        assert score.field_accuracy["position"] < 1.0

    def test_omission_and_hallucination_rates(self):
        gt = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        pred = [{"name": "A"}, {"name": "D"}]

        score = score_entity_list(gt, pred, ("name",))

        assert score.omission_rate > 0
        assert score.hallucination_rate > 0

    def test_quality_weighted_wrong_dates_lower(self):
        """A match with wrong dates scores lower than a perfect match."""
        gt = [{"company": "Google", "position": "SWE", "startYear": 2020}]

        perfect_pred = [{"company": "Google", "position": "SWE", "startYear": 2020}]
        wrong_date_pred = [{"company": "Google", "position": "SWE", "startYear": 2019}]

        perfect_score = score_entity_list(gt, perfect_pred, ("company", "position"))
        wrong_score = score_entity_list(gt, wrong_date_pred, ("company", "position"))

        assert perfect_score.f1 == 1.0
        assert wrong_score.f1 < 1.0

    def test_quality_weighted_empty_gt_fields_not_penalised(self):
        """Fields absent from GT should not affect quality."""
        gt = [{"company": "Google"}]
        pred = [{"company": "Google", "startYear": 2020, "position": "SWE"}]

        score = score_entity_list(gt, pred, ("company",))

        # GT only has 'company' → quality based on company alone → 1.0
        assert score.f1 == 1.0

    def test_quality_weighted_description_affects_f1(self):
        """Wrong descriptions should pull F1 below 1.0."""
        gt = [{"company": "Google", "description": ["Built search engine", "Led team of 5"]}]
        pred = [{"company": "Google", "description": ["Unrelated work on something else"]}]

        score = score_entity_list(gt, pred, ("company",))

        # Company matches perfectly but description is wrong → quality < 1.0
        assert score.f1 < 1.0


class TestEntityQuality:

    def test_perfect_match(self):
        gt = {"company": "Google", "position": "SWE", "startYear": 2020}
        pred = {"company": "Google", "position": "SWE", "startYear": 2020}

        assert _entity_quality(gt, pred) == 1.0

    def test_empty_gt_fields_skipped(self):
        gt = {"company": "Google", "position": "", "startYear": None}
        pred = {"company": "Google"}

        # Only 'company' has GT data → quality based on company alone
        assert _entity_quality(gt, pred) == 1.0

    def test_wrong_integer_field(self):
        gt = {"company": "Google", "startYear": 2020}
        pred = {"company": "Google", "startYear": 2019}

        q = _entity_quality(gt, pred)
        # company=1.0, startYear=0.0 → avg = 0.5
        assert q == 0.5

    def test_boolean_field(self):
        gt = {"company": "Google", "inProgress": True}
        pred = {"company": "Google", "inProgress": False}

        q = _entity_quality(gt, pred)
        # company=1.0, inProgress=0.0 → avg = 0.5
        assert q == 0.5

    def test_empty_list_skipped(self):
        gt = {"company": "Google", "description": []}
        pred = {"company": "Google"}

        assert _entity_quality(gt, pred) == 1.0

    def test_description_positional(self):
        gt = {"company": "Google", "description": ["Built search features", "Led team"]}
        pred = {"company": "Google", "description": ["Built search features", "Led team"]}

        assert _entity_quality(gt, pred) == 1.0

    def test_description_reordered_lower(self):
        gt = {"company": "Google", "description": ["Built search features", "Led team"]}
        pred = {"company": "Google", "description": ["Led team", "Built search features"]}

        q = _entity_quality(gt, pred)
        # company=1.0, description < 1.0 → avg < 1.0
        assert q < 1.0


class TestPositionalBulletScore:

    def test_perfect_match(self):
        assert _positional_bullet_score(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_both_empty(self):
        assert _positional_bullet_score([], []) == 1.0

    def test_missing_bullet(self):
        score = _positional_bullet_score(["A", "B", "C"], ["A", "B"])
        # (1.0 + 1.0 + 0.0) / 3 ≈ 0.667
        assert abs(score - 2 / 3) < 0.01

    def test_extra_bullet(self):
        score = _positional_bullet_score(["A"], ["A", "Hallucinated"])
        # (1.0 + 0.0) / 2 = 0.5
        assert score == 0.5

    def test_reordered_bullets(self):
        score = _positional_bullet_score(
            ["Built search features", "Led team of 5"],
            ["Led team of 5", "Built search features"],
        )
        assert score < 1.0

    def test_similar_bullets(self):
        score = _positional_bullet_score(
            ["Built search features for Google"],
            ["Built search feature for Google"],
        )
        assert score > 0.9

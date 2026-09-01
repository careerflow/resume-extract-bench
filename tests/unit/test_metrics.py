from resume_bench.grading.metrics import score_entity_list, score_flat_list, score_singleton


class TestScoreSingleton:

    def test_perfect_match(self):
        gt = {"fname": "John", "lname": "Doe"}
        pred = {"fname": "John", "lname": "Doe"}

        score = score_singleton(gt, pred, ("fname", "lname"))

        assert score.f1 == 1.0

    def test_partial_match(self):
        gt = {"fname": "John", "lname": "Doe", "email": "john@test.com"}
        pred = {"fname": "John", "lname": "Doe", "email": ""}

        score = score_singleton(gt, pred, ("fname", "lname", "email"))

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

        assert score.field_accuracy["fname"] == 1.0
        assert score.field_accuracy["lname"] < 1.0


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
        assert score.description_token_f1 > 0.9

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

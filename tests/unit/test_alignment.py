from resume_bench.grading.alignment import align_entities


class TestAlignEntities:

    def test_empty_gt(self):
        matched, missed, spurious = align_entities([], [{"name": "A"}], ["name"])
        assert matched == []
        assert missed == []
        assert spurious == [0]

    def test_empty_pred(self):
        matched, missed, spurious = align_entities([{"name": "A"}], [], ["name"])
        assert matched == []
        assert missed == [0]
        assert spurious == []

    def test_both_empty(self):
        matched, missed, spurious = align_entities([], [], ["name"])
        assert matched == []
        assert missed == []
        assert spurious == []

    def test_perfect_match(self):
        gt = [{"name": "Python"}, {"name": "JavaScript"}]
        pred = [{"name": "Python"}, {"name": "JavaScript"}]

        matched, missed, spurious = align_entities(gt, pred, ["name"])

        assert len(matched) == 2
        assert missed == []
        assert spurious == []

    def test_case_insensitive_matching(self):
        gt = [{"name": "Python"}]
        pred = [{"name": "python"}]

        matched, missed, spurious = align_entities(gt, pred, ["name"])

        assert len(matched) == 1

    def test_below_threshold(self):
        gt = [{"name": "Python"}]
        pred = [{"name": "Java"}]

        matched, missed, spurious = align_entities(gt, pred, ["name"], threshold=0.9)

        assert len(matched) == 0
        assert missed == [0]
        assert spurious == [0]

    def test_multi_field_alignment(self):
        gt = [{"company": "Google", "position": "SWE"}]
        pred = [{"company": "Google", "position": "Software Engineer"}]

        matched, missed, spurious = align_entities(gt, pred, ["company", "position"])

        assert len(matched) == 1

    def test_unequal_lengths(self):
        gt = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        pred = [{"name": "A"}, {"name": "B"}]

        matched, missed, spurious = align_entities(gt, pred, ["name"])

        assert len(matched) == 2
        assert len(missed) == 1
        assert spurious == []

    def test_more_pred_than_gt(self):
        gt = [{"name": "A"}]
        pred = [{"name": "A"}, {"name": "B"}, {"name": "C"}]

        matched, missed, spurious = align_entities(gt, pred, ["name"])

        assert len(matched) == 1
        assert missed == []
        assert len(spurious) == 2

from resume_bench.grading.text import field_similarity, normalize_text, token_f1


class TestNormalizeText:

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_lowercases(self):
        assert normalize_text("HELLO") == "hello"

    def test_strips_llc(self):
        assert normalize_text("Google LLC") == "google"

    def test_strips_inc_with_period(self):
        assert normalize_text("Apple Inc.") == "apple"

    def test_strips_corp(self):
        assert normalize_text("Amazon Corp") == "amazon"

    def test_strips_ltd(self):
        assert normalize_text("Samsung Ltd.") == "samsung"

    def test_strips_gmbh(self):
        assert normalize_text("Siemens GmbH") == "siemens"

    def test_no_suffix_unchanged(self):
        assert normalize_text("Stanford University") == "stanford university"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestFieldSimilarity:

    def test_both_empty(self):
        assert field_similarity("", "") == 1.0

    def test_both_none(self):
        assert field_similarity(None, None) == 1.0

    def test_one_empty(self):
        assert field_similarity("hello", "") == 0.0

    def test_one_none(self):
        assert field_similarity(None, "hello") == 0.0

    def test_identical(self):
        assert field_similarity("Google", "Google") == 1.0

    def test_case_insensitive(self):
        assert field_similarity("Google", "google") == 1.0

    def test_corporate_suffix_ignored(self):
        sim = field_similarity("Google LLC", "Google")
        assert sim == 1.0

    def test_similar_strings(self):
        sim = field_similarity("Stanford University", "Standford University")
        assert sim > 0.9

    def test_dissimilar_strings(self):
        sim = field_similarity("Google", "Microsoft")
        assert sim < 0.7


class TestTokenF1:

    def test_perfect_match(self):
        assert token_f1("the quick brown fox", "the quick brown fox") == 1.0

    def test_partial_overlap(self):
        score = token_f1("the quick brown fox", "the quick fox")
        assert 0.7 < score < 1.0

    def test_no_overlap(self):
        assert token_f1("hello world", "foo bar") == 0.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_gt_empty(self):
        assert token_f1("", "hello world") == 0.0

    def test_pred_empty(self):
        assert token_f1("hello world", "") == 0.0

    def test_superset_prediction(self):
        score = token_f1("a b", "a b c d")
        assert score < 1.0
        assert score > 0.5

    def test_subset_prediction(self):
        score = token_f1("a b c d", "a b")
        assert score < 1.0
        assert score > 0.5

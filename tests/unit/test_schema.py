from resume_bench.schema import RESUME_SCHEMA, get_schema
from resume_bench.schema.sections import SECTIONS, SectionKind


class TestSchema:

    def test_schema_loads(self):
        schema = get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_all_sections_in_schema(self):
        schema = get_schema()
        props = schema["properties"]

        for spec in SECTIONS:
            assert spec.name in props, f"{spec.name} missing from schema"

    def test_get_schema_returns_copy(self):
        a = get_schema()
        b = get_schema()
        a["test"] = True
        assert "test" not in b

    def test_resume_schema_constant(self):
        assert RESUME_SCHEMA is not None
        assert "properties" in RESUME_SCHEMA


class TestSections:

    def test_section_count(self):
        assert len(SECTIONS) == 9

    def test_basics_is_singleton(self):
        basics = next(s for s in SECTIONS if s.name == "basics")
        assert basics.kind == SectionKind.SINGLETON

    def test_skills_is_flat_list(self):
        skills = next(s for s in SECTIONS if s.name == "skills")
        assert skills.kind == SectionKind.FLAT_LIST

    def test_experience_has_description_scoring(self):
        exp = next(s for s in SECTIONS if s.name == "experience")
        assert exp.score_description is True

    def test_certifications_no_description_scoring(self):
        certs = next(s for s in SECTIONS if s.name == "certifications")
        assert certs.score_description is False

    def test_all_section_names_unique(self):
        names = [s.name for s in SECTIONS]
        assert len(names) == len(set(names))

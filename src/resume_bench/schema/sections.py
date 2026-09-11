"""Section specifications for resume extraction scoring.

This module defines the structure and metadata for each resume section,
including how to identify key fields and whether to score description arrays.

Two variants are provided:
- ``SECTIONS_FULL``: scores all schema fields (matches ExtractBench conventions).
- ``SECTIONS_TRIMMED``: drops fields with 0% ground-truth coverage to avoid
  false hallucination penalties (matches internal benchmark behaviour).

Use ``get_sections()`` to pick the right variant based on
``settings.use_trimmed_schema``.
"""

from dataclasses import dataclass
from enum import Enum


class SectionKind(str, Enum):
    """Type of section structure in the resume schema."""

    SINGLETON = "singleton"
    FLAT_LIST = "flat_list"
    ENTITY_LIST = "entity_list"


@dataclass(frozen=True)
class SectionSpec:
    """Specification for a resume section.

    Attributes:
        name: Section name as it appears in the schema.
        kind: Structure type of the section.
        key_fields: Tuple of field names used to identify unique entities.
        score_description: Whether to score the description array for this section.
    """

    name: str
    kind: SectionKind
    key_fields: tuple[str, ...]
    score_description: bool = False


# ---------------------------------------------------------------------------
# Full section specs — all schema fields scored
# ---------------------------------------------------------------------------
SECTIONS_FULL: tuple[SectionSpec, ...] = (
    SectionSpec(
        "basics",
        SectionKind.SINGLETON,
        ("fname", "lname", "email", "phone", "city", "state", "country", "hasPersonalPhoto")
    ),
    SectionSpec(
        "experience",
        SectionKind.ENTITY_LIST,
        ("company", "position"),
        score_description=True
    ),
    SectionSpec(
        "education",
        SectionKind.ENTITY_LIST,
        ("institution", "area"),
        score_description=True
    ),
    SectionSpec(
        "projects",
        SectionKind.ENTITY_LIST,
        ("name",),
        score_description=True
    ),
    SectionSpec(
        "personalSummary",
        SectionKind.ENTITY_LIST,
        ("text",)
    ),
    SectionSpec(
        "certifications",
        SectionKind.ENTITY_LIST,
        ("name", "issuer")
    ),
    SectionSpec(
        "awards",
        SectionKind.ENTITY_LIST,
        ("title",)
    ),
    SectionSpec(
        "volunteering",
        SectionKind.ENTITY_LIST,
        ("organization", "position"),
        score_description=True
    ),
    SectionSpec(
        "skills",
        SectionKind.FLAT_LIST,
        ()
    ),
)

# ---------------------------------------------------------------------------
# Trimmed section specs — drops fields with 0% GT coverage
#
# Removed fields:
#   basics: country
#
# Note: the other trimmed fields from the internal repo (experience.country,
# education.startMonth/endMonth/currentlyStudyHere/city/country,
# projects.url, volunteering.startYear/endYear/currentlyVolunteerHere)
# are not key_fields in either variant, so they don't affect scoring.
# ---------------------------------------------------------------------------
SECTIONS_TRIMMED: tuple[SectionSpec, ...] = (
    SectionSpec(
        "basics",
        SectionKind.SINGLETON,
        ("fname", "lname", "email", "phone", "city", "state", "hasPersonalPhoto")
    ),
    SectionSpec(
        "experience",
        SectionKind.ENTITY_LIST,
        ("company", "position"),
        score_description=True
    ),
    SectionSpec(
        "education",
        SectionKind.ENTITY_LIST,
        ("institution", "area"),
        score_description=True
    ),
    SectionSpec(
        "projects",
        SectionKind.ENTITY_LIST,
        ("name",),
        score_description=True
    ),
    SectionSpec(
        "personalSummary",
        SectionKind.ENTITY_LIST,
        ("text",)
    ),
    SectionSpec(
        "certifications",
        SectionKind.ENTITY_LIST,
        ("name", "issuer")
    ),
    SectionSpec(
        "awards",
        SectionKind.ENTITY_LIST,
        ("title",)
    ),
    SectionSpec(
        "volunteering",
        SectionKind.ENTITY_LIST,
        ("organization", "position"),
        score_description=True
    ),
    SectionSpec(
        "skills",
        SectionKind.FLAT_LIST,
        ()
    ),
)


def get_sections() -> tuple[SectionSpec, ...]:
    """Return section specs based on ``settings.use_trimmed_schema``."""
    from resume_bench.settings import settings

    return SECTIONS_TRIMMED if settings.use_trimmed_schema else SECTIONS_FULL


# Backwards-compatible alias — existing code imports ``SECTIONS``.
SECTIONS = SECTIONS_TRIMMED

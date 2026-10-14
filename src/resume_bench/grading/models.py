from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SectionScore:
    gt_count: int = 0
    pred_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    omission_rate: float = 0.0
    hallucination_rate: float = 0.0
    field_accuracy: dict[str, float] = field(default_factory=dict)
    description_token_f1: float | None = None
    is_vacuous: bool = False
    in_headline: bool = True


@dataclass
class ResumeScore:
    resume_id: str = ""
    sections: dict[str, SectionScore] = field(default_factory=dict)
    entity_f1: float = 0.0
    completed: bool = True

    @property
    def non_vacuous_sections(self) -> dict[str, SectionScore]:
        return {k: v for k, v in self.sections.items() if not v.is_vacuous}

    @property
    def headline_sections(self) -> dict[str, SectionScore]:
        return {
            k: v for k, v in self.sections.items()
            if not v.is_vacuous and v.in_headline
        }

    @property
    def macro_entity_f1(self) -> float:
        hl = self.headline_sections

        if not hl:
            return 0.0

        return sum(s.f1 for s in hl.values()) / len(hl)

    @property
    def basics_field_accuracy(self) -> float:
        basics = self.sections.get("basics")

        if not basics or not basics.field_accuracy:
            return 0.0

        vals = basics.field_accuracy.values()
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class CellCounts:
    """Cell-level counts for ExtractBench-exact scoring."""

    correct: int = 0
    expected: int = 0
    predicted: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


@dataclass
class ExtractBenchScore:
    """Per-resume result from ExtractBench-exact scoring."""

    resume_id: str = ""
    cells: CellCounts = field(default_factory=CellCounts)
    completed: bool = True


@dataclass
class GradingConfig:
    threshold: float = 0.5
    score_all_fields: bool = False

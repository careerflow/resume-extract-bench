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
    def macro_entity_f1(self) -> float:
        nv = self.non_vacuous_sections

        if not nv:
            return 0.0

        return sum(s.f1 for s in nv.values()) / len(nv)


@dataclass
class GradingConfig:
    threshold: float = 0.5

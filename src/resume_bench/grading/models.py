from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CellCounts:
    """Cell-level counts for ResumeBench scoring."""

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
class ResumeBenchScore:
    """Per-resume result from ResumeBench-exact scoring."""

    resume_id: str = ""
    cells: CellCounts = field(default_factory=CellCounts)
    completed: bool = True

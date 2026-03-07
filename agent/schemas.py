from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RewriteSpec:
    user_prompt: str
    poem_type: str
    script: str
    style: str = "典雅含蓄"
    focus: str = ""
    constraints: list[str] = field(default_factory=list)
    variant: Optional[str] = None
    rhyme_group: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanSpec:
    outline: list[str] = field(default_factory=list)
    imagery: list[str] = field(default_factory=list)
    emotional_arc: str = ""
    closing: str = ""
    cautions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    format_ok: bool = False
    line_length_ok: bool = False
    rhyming_ok: bool = False
    meter_ok: bool = False
    passed: bool = False
    line_lengths: list[int] = field(default_factory=list)
    expected_line_lengths: list[int] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DraftAttempt:
    round_index: int
    candidate_index: int
    poem_text: str
    evaluation_report: EvaluationReport
    repair_hint_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluation_report"] = self.evaluation_report.to_dict()
        return data


@dataclass
class AgentResult:
    success: bool
    poem_text: str
    rewrite_spec: RewriteSpec
    plan_spec: PlanSpec
    evaluation_report: EvaluationReport
    attempts: list[DraftAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "poem_text": self.poem_text,
            "rewrite_spec": self.rewrite_spec.to_dict(),
            "plan_spec": self.plan_spec.to_dict(),
            "evaluation_report": self.evaluation_report.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

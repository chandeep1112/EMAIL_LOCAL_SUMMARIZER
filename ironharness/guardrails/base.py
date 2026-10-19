"""Shared types for guardrail evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    rule_id: str
    type: str
    action: str                 # what the policy said to do
    score: float = 1.0
    # Human-readable *categories only*. A finding is written to the audit log and
    # the dashboard, so it must never carry the matched text: naming the secret
    # you caught is the same leak you were hired to prevent.
    detail: List[str] = field(default_factory=list)
    message: str = ""
    segment: int = -1
    direction: str = "input"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule_id, "type": self.type, "action": self.action,
            "score": self.score, "detail": self.detail, "message": self.message,
            "segment": self.segment, "direction": self.direction,
        }


@dataclass
class Outcome:
    segments: List[str]
    findings: List[Finding] = field(default_factory=list)
    blocked: Optional[Finding] = None
    changed: bool = False

    @property
    def is_blocked(self) -> bool:
        return self.blocked is not None

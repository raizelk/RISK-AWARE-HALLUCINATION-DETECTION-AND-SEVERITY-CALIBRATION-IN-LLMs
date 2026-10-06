"""Stable records shared by dataset, retrieval, verification, and evaluation."""

from dataclasses import dataclass, field
from enum import StrEnum


class VerificationLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    text: str
    start: int = 0
    end: int = 0


@dataclass(frozen=True, slots=True)
class VerificationResult:
    claim: Claim
    label: VerificationLabel
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    similarity: float = 0.0
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model_name: str
    dataset_item_id: str
    text: str
    confidence: float | None = None

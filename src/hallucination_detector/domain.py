"""Stable records shared by dataset, retrieval, verification, and evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python < 3.11
    class StrEnum(str, Enum):
        pass


class VerificationLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class VerificationEvidenceScore:
    """Signals recorded for one evidence passage used to verify a claim."""

    evidence: Evidence
    retrieval_score: float
    similarity: float
    entailment_probability: float
    contradiction_probability: float
    neutral_probability: float
    support_score: float
    contradiction_score: float


@dataclass(frozen=True)
class VerificationResult:
    claim: Claim
    label: VerificationLabel
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    similarity: float = 0.0
    explanation: str = ""
    support_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    evidence_scores: tuple[VerificationEvidenceScore, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelResponse:
    model_name: str
    dataset_item_id: str
    text: str
    confidence: Optional[float] = None

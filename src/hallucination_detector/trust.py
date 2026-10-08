"""Risk-aware, multi-signal trust scoring for model responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .domain import Severity, VerificationLabel, VerificationResult
from .severity import SeverityAssessment

_SEVERITY_PENALTIES = {
    Severity.MILD: 0.05,
    Severity.MODERATE: 0.15,
    Severity.SEVERE: 0.30,
    Severity.CRITICAL: 0.60,
}


@dataclass(frozen=True)
class TrustScore:
    """A bounded score and its component signals for one model response."""

    score: float
    factual_score: float
    semantic_alignment: float
    consistency_score: float
    confidence_calibration: Optional[float]
    severity_penalty: float
    claim_count: int
    rationale: str


class TrustScorer:
    """Combine factual, semantic, consistency, and confidence signals.

    The defaults are transparent policy weights, not learned probabilities.
    They should be tuned against a labeled evaluation set before research
    conclusions are drawn.
    """

    def __init__(
        self,
        *,
        factual_weight: float = 0.45,
        semantic_weight: float = 0.20,
        consistency_weight: float = 0.20,
        confidence_weight: float = 0.15,
    ) -> None:
        weights = (
            factual_weight,
            semantic_weight,
            consistency_weight,
            confidence_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Trust weights must be finite and non-negative.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Trust weights must sum to 1.")
        self._weights = weights

    def score(
        self,
        verifications: Sequence[VerificationResult],
        *,
        severity_assessments: Sequence[SeverityAssessment] = (),
        consistency_score: float = 1.0,
        confidences: Sequence[Optional[float]] = (),
    ) -> TrustScore:
        """Score one response from claim verification and optional signals.

        ``consistency_score`` is supplied by a repeated/paraphrased-response
        evaluator and must already be normalized to [0, 1]. ``confidences``
        must align with verification order; omitted values leave confidence
        calibration unavailable rather than silently treating it as perfect.
        """
        results = tuple(verifications)
        if not results:
            raise ValueError("At least one verification result is required.")
        _validate_unit_interval(consistency_score, "consistency_score")
        if confidences and len(confidences) != len(results):
            raise ValueError("Confidences must contain one value per verification result.")
        for confidence in confidences:
            if confidence is not None:
                _validate_unit_interval(confidence, "confidence")

        factual_score = sum(
            result.label is VerificationLabel.SUPPORTED for result in results
        ) / len(results)
        semantic_alignment = sum(
            _validate_signal(result.similarity, "similarity") for result in results
        ) / len(results)
        calibration = self._confidence_calibration(results, confidences)
        severity_penalty = self._severity_penalty(
            results, tuple(severity_assessments)
        )

        confidence_signal = calibration if calibration is not None else 0.0
        score = (
            self._weights[0] * factual_score
            + self._weights[1] * semantic_alignment
            + self._weights[2] * consistency_score
            + self._weights[3] * confidence_signal
            - severity_penalty
        )
        score = max(0.0, min(1.0, score))
        rationale = (
            f"Factual={factual_score:.3f}; semantic={semantic_alignment:.3f}; "
            f"consistency={consistency_score:.3f}; "
            f"confidence calibration="
            f"{'unavailable' if calibration is None else f'{calibration:.3f}'}; "
            f"severity penalty={severity_penalty:.3f}."
        )
        return TrustScore(
            score=score,
            factual_score=factual_score,
            semantic_alignment=semantic_alignment,
            consistency_score=consistency_score,
            confidence_calibration=calibration,
            severity_penalty=severity_penalty,
            claim_count=len(results),
            rationale=rationale,
        )

    @staticmethod
    def _confidence_calibration(
        results: tuple[VerificationResult, ...],
        confidences: Sequence[Optional[float]],
    ) -> Optional[float]:
        if not confidences or all(confidence is None for confidence in confidences):
            return None
        calibration_values = []
        for result, confidence in zip(results, confidences):
            if confidence is None:
                continue
            correctness = 1.0 if result.label is VerificationLabel.SUPPORTED else 0.0
            calibration_values.append(1.0 - abs(confidence - correctness))
        return sum(calibration_values) / len(calibration_values)

    @staticmethod
    def _severity_penalty(
        results: tuple[VerificationResult, ...],
        assessments: tuple[SeverityAssessment, ...],
    ) -> float:
        by_claim = {assessment.verification.claim.id: assessment for assessment in assessments}
        if len(by_claim) != len(assessments):
            raise ValueError("Severity assessments must have unique claim IDs.")
        penalty = 0.0
        for result in results:
            assessment = by_claim.get(result.claim.id)
            if assessment is None:
                continue
            if assessment.verification != result:
                raise ValueError(
                    f"Severity assessment for claim {result.claim.id!r} does not match."
                )
            if assessment.severity is not None:
                penalty += _SEVERITY_PENALTIES[assessment.severity]
        return min(1.0, penalty / len(results))


def _validate_signal(value: float, name: str) -> float:
    _validate_unit_interval(value, name)
    return value


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")

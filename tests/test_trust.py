import pytest

from hallucination_detector.domain import Claim, Severity, VerificationLabel, VerificationResult
from hallucination_detector.severity import SeverityAssessment
from hallucination_detector.trust import TrustScorer


def _result(claim_id, label, similarity):
    return VerificationResult(
        claim=Claim(claim_id, f"claim {claim_id}"),
        label=label,
        similarity=similarity,
    )


def test_trust_score_combines_signals_and_preserves_components():
    results = (
        _result("supported", VerificationLabel.SUPPORTED, 0.9),
        _result("unknown", VerificationLabel.UNKNOWN, 0.4),
    )

    score = TrustScorer().score(
        results,
        consistency_score=0.8,
        confidences=(0.9, 0.2),
    )

    assert score.claim_count == 2
    assert score.factual_score == 0.5
    assert score.semantic_alignment == pytest.approx(0.65)
    assert score.confidence_calibration == pytest.approx(0.85)
    assert score.severity_penalty == 0
    assert score.score == pytest.approx(0.6425)


def test_severity_penalty_is_risk_aware_and_bounded():
    result = _result("critical", VerificationLabel.CONTRADICTED, 0.2)
    verification = SeverityAssessment(
        verification=result,
        severity=Severity.CRITICAL,
        risk_domains=("medical",),
        rationale="Unsafe advice.",
    )

    score = TrustScorer().score(
        (result,),
        severity_assessments=(verification,),
        confidences=(0.95,),
    )

    assert score.severity_penalty == pytest.approx(0.6)
    assert score.score == 0


def test_confidence_is_unavailable_when_not_provided():
    score = TrustScorer().score(
        (_result("claim-1", VerificationLabel.SUPPORTED, 1.0),)
    )

    assert score.confidence_calibration is None
    assert "unavailable" in score.rationale


@pytest.mark.parametrize(
    "kwargs",
    (
        {"consistency_score": -0.1},
        {"consistency_score": 1.1},
        {"confidences": (0.5, 0.5)},
    ),
)
def test_score_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        TrustScorer().score(
            (_result("claim-1", VerificationLabel.SUPPORTED, 0.5),),
            **kwargs,
        )


def test_scorer_rejects_invalid_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        TrustScorer(factual_weight=0.5, semantic_weight=0.5, consistency_weight=0.5)

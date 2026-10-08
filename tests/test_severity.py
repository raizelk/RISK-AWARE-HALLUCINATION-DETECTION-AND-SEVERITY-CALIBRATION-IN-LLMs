import pytest

from hallucination_detector.domain import Claim, Severity, VerificationLabel, VerificationResult
from hallucination_detector.severity import SeverityClassifier


def _verification(
    label=VerificationLabel.UNKNOWN,
    *,
    text="The museum opened in 1950.",
    contradiction_score=0.0,
):
    return VerificationResult(
        claim=Claim("claim-1", text),
        label=label,
        contradiction_score=contradiction_score,
    )


def test_supported_claim_has_no_error_severity():
    result = SeverityClassifier().classify(_verification(VerificationLabel.SUPPORTED))

    assert result.severity is None
    assert result.rationale == "The claim is supported; no error severity is assigned."


def test_unknown_claim_defaults_to_mild_and_contradiction_to_moderate():
    classifier = SeverityClassifier()

    assert classifier.classify(_verification()).severity is Severity.MILD
    assert (
        classifier.classify(_verification(VerificationLabel.CONTRADICTED)).severity
        is Severity.MODERATE
    )


def test_strong_contradiction_and_high_centrality_escalate_severity():
    classifier = SeverityClassifier()

    strong_contradiction = classifier.classify(
        _verification(
            VerificationLabel.CONTRADICTED,
            contradiction_score=0.9,
        )
    )
    central_claim = classifier.classify(
        _verification(VerificationLabel.CONTRADICTED),
        centrality=0.8,
    )

    assert strong_contradiction.severity is Severity.SEVERE
    assert central_claim.severity is Severity.SEVERE


def test_ordinary_claim_cannot_reach_critical_from_centrality_and_confidence_alone():
    result = SeverityClassifier().classify(
        _verification(
            VerificationLabel.CONTRADICTED,
            contradiction_score=0.95,
        ),
        centrality=1.0,
    )

    assert result.severity is Severity.SEVERE


def test_central_high_impact_claim_is_severe():
    result = SeverityClassifier().classify(
        _verification(text="The court ruling changed the legal rights of tenants."),
        centrality=0.6,
    )

    assert result.severity is Severity.SEVERE
    assert result.risk_domains == ("legal",)
    assert "high-impact domain" in result.rationale


@pytest.mark.parametrize(
    "text,domain",
    (
        ("You should double your medication dose without consulting a doctor.", "medical"),
        ("You should invest your retirement fund in this single stock.", "finance"),
        ("You must sign this contract immediately.", "legal"),
        ("You should mix these chemicals in an enclosed room.", "safety"),
    ),
)
def test_unsupported_actionable_high_impact_advice_is_critical(text, domain):
    result = SeverityClassifier().classify(_verification(text=text))

    assert result.severity is Severity.CRITICAL
    assert result.risk_domains == (domain,)
    assert "actionable advice" in result.rationale


def test_high_impact_domain_without_advice_is_not_automatically_critical():
    result = SeverityClassifier().classify(
        _verification(
            VerificationLabel.CONTRADICTED,
            text="The court issued its ruling on Monday.",
        )
    )

    assert result.severity is Severity.MODERATE
    assert result.risk_domains == ("legal",)


@pytest.mark.parametrize("centrality", (-0.1, 1.1, float("nan"), float("inf")))
def test_classifier_rejects_invalid_centrality(centrality):
    with pytest.raises(ValueError, match="centrality"):
        SeverityClassifier().classify(_verification(), centrality=centrality)


@pytest.mark.parametrize("score", (-0.1, 1.1, float("nan"), float("inf")))
def test_classifier_rejects_invalid_contradiction_scores(score):
    with pytest.raises(ValueError, match="contradiction_score"):
        SeverityClassifier().classify(
            _verification(VerificationLabel.CONTRADICTED, contradiction_score=score)
        )


def test_classifier_rejects_invalid_high_contradiction_threshold():
    with pytest.raises(ValueError, match="high_contradiction_threshold"):
        SeverityClassifier(high_contradiction_threshold=1.1)

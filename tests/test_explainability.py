import pytest

from hallucination_detector.domain import (
    Claim,
    Evidence,
    Severity,
    VerificationLabel,
    VerificationResult,
)
from hallucination_detector.explainability import explain_response
from hallucination_detector.severity import SeverityAssessment


def test_explain_response_highlights_claim_and_includes_evidence_and_reasons():
    response = "The moon is made of cheese."
    claim = Claim("claim-1", response, start=0, end=len(response))
    evidence = Evidence("source-1", "The moon is a natural satellite of Earth.", "reference")
    verification = VerificationResult(
        claim=claim,
        label=VerificationLabel.CONTRADICTED,
        evidence=(evidence,),
        explanation="The retrieved evidence contradicts the claim.",
    )
    severity = SeverityAssessment(
        verification=verification,
        severity=Severity.MODERATE,
        risk_domains=(),
        rationale="Contradicted claims start at moderate severity.",
    )

    result = explain_response(response, (verification,), (severity,))

    assert result.original_response == response
    assert result.highlighted_response_html == (
        '<mark data-claim-id="claim-1" data-verification="contradicted" '
        'data-severity="moderate">The moon is made of cheese.</mark>'
    )
    assert result.claims[0].evidence == (evidence,)
    assert result.claims[0].verification_reason == verification.explanation
    assert result.claims[0].severity is Severity.MODERATE
    assert result.claims[0].severity_reason == severity.rationale


def test_explain_response_highlights_multiple_claims_in_original_order():
    response = "Canberra is the capital. Sydney is in Australia."
    first = Claim("claim-1", "Canberra is the capital.", 0, 24)
    second_text = "Sydney is in Australia."
    second_start = response.index(second_text)
    second = Claim("claim-2", second_text, second_start, second_start + len(second_text))
    verifications = (
        VerificationResult(first, VerificationLabel.SUPPORTED),
        VerificationResult(second, VerificationLabel.UNKNOWN),
    )

    result = explain_response(response, verifications)

    assert [item.claim.id for item in result.claims] == ["claim-1", "claim-2"]
    assert result.highlighted_response_html == (
        '<mark data-claim-id="claim-1" data-verification="supported">'
        "Canberra is the capital.</mark> "
        '<mark data-claim-id="claim-2" data-verification="unknown">'
        "Sydney is in Australia.</mark>"
    )


def test_highlighted_response_html_escapes_untrusted_text_and_attributes():
    response = '<script>alert("bad")</script>'
    claim = Claim('claim-"1', response, 0, len(response))
    verification = VerificationResult(claim, VerificationLabel.UNKNOWN)

    result = explain_response(response, (verification,))

    assert "<script>" not in result.highlighted_response_html
    assert "&lt;script&gt;" in result.highlighted_response_html
    assert 'data-claim-id="claim-&quot;1"' in result.highlighted_response_html


@pytest.mark.parametrize(
    "claim",
    (
        Claim("negative", "word", -1, 3),
        Claim("outside", "word", 0, 5),
        Claim("empty", "", 1, 1),
        Claim("mismatch", "other", 0, 4),
    ),
)
def test_explain_response_rejects_invalid_claim_spans(claim):
    with pytest.raises(ValueError, match="span"):
        explain_response("word", (VerificationResult(claim, VerificationLabel.UNKNOWN),))


def test_explain_response_rejects_overlapping_claim_spans():
    response = "abcdef"
    first = VerificationResult(
        Claim("claim-1", "abcd", 0, 4),
        VerificationLabel.UNKNOWN,
    )
    second = VerificationResult(
        Claim("claim-2", "cdef", 2, 6),
        VerificationLabel.CONTRADICTED,
    )

    with pytest.raises(ValueError, match="overlap"):
        explain_response(response, (first, second))


def test_explain_response_rejects_severity_for_nonmatching_verification():
    response = "Canberra is the capital."
    claim = Claim("claim-1", response, 0, len(response))
    verification = VerificationResult(claim, VerificationLabel.UNKNOWN)
    mismatched = SeverityAssessment(
        verification=VerificationResult(claim, VerificationLabel.CONTRADICTED),
        severity=Severity.MODERATE,
        risk_domains=(),
        rationale="mismatched result",
    )

    with pytest.raises(ValueError, match="does not match"):
        explain_response(response, (verification,), (mismatched,))


def test_explain_response_escapes_unhighlighted_text():
    response = "<b>Answer:</b> Canberra is the capital."
    claim_text = "Canberra is the capital."
    start = response.index(claim_text)
    verification = VerificationResult(
        Claim("claim-1", claim_text, start, start + len(claim_text)),
        VerificationLabel.SUPPORTED,
    )

    result = explain_response(response, (verification,))

    assert result.highlighted_response_html.startswith("&lt;b&gt;Answer:&lt;/b&gt; ")

from hallucination_detector.baseline import extract_claims, lexical_similarity, verify_claim
from hallucination_detector.domain import Evidence, VerificationLabel


def test_extract_claims_preserves_response_offsets():
    claims = extract_claims("Water freezes at 0 C. The moon is a planet!")

    assert [claim.text for claim in claims] == ["Water freezes at 0 C.", "The moon is a planet!"]
    assert claims[0].start == 0
    assert claims[1].text == "The moon is a planet!"


def test_lexical_similarity_is_zero_for_disjoint_text():
    assert lexical_similarity("cats", "quantum mechanics") == 0.0


def test_verify_claim_returns_supported_for_matching_evidence():
    result = verify_claim(
        extract_claims("Canberra is the capital of Australia.")[0],
        (Evidence("a1", "Canberra is the capital city of Australia."),),
    )

    assert result.label is VerificationLabel.SUPPORTED
    assert result.evidence[0].id == "a1"


def test_verify_claim_returns_unknown_without_matching_evidence():
    result = verify_claim(
        extract_claims("The moon is made of cheese.")[0],
        (Evidence("a1", "The moon orbits Earth."),),
    )

    assert result.label is VerificationLabel.UNKNOWN
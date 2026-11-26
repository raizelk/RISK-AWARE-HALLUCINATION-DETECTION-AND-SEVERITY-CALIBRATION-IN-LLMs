from hallucination_detector.baseline import extract_claims, lexical_similarity, verify_claim
from hallucination_detector.domain import Evidence, VerificationLabel


def test_extract_claims_preserves_response_offsets():
    claims = extract_claims("Water freezes at 0 C. The moon is a planet!")

    assert [claim.text for claim in claims] == ["Water freezes at 0 C.", "The moon is a planet!"]
    assert claims[0].start == 0
    assert claims[1].text == "The moon is a planet!"


def test_extract_claims_splits_multi_clause_sentences():
    claims = extract_claims(
        "Canberra is the capital of Australia and Sydney is the capital of New South Wales."
    )

    assert [claim.text for claim in claims] == [
        "Canberra is the capital of Australia",
        "Sydney is the capital of New South Wales.",
    ]
    assert claims[0].start == 0
    assert claims[1].start > claims[0].start


def test_extract_claims_skips_introductions_headings_and_topic_fragments():
    response = (
        "Here are the key purposes and benefits of vaccination:\n"
        "1. **Prevention of Disease**: Vaccines protect people. "
        "**Benefits of vaccination**. Reducing morbidity and mortality."
    )

    claims = extract_claims(response)

    assert [claim.text for claim in claims] == ["Vaccines protect people."]
    assert response[claims[0].start : claims[0].end] == claims[0].text


def test_extract_claims_rejects_incomplete_clauses_and_modifier_fragments():
    response = (
        "Vaccines work by introducing a weakened or dead form of a pathogen. "
        "Vaccines work by introducing a weakened. Reducing morbidity and mortality."
    )

    claims = extract_claims(response)

    assert [claim.text for claim in claims] == [
        "Vaccines work by introducing a weakened or dead form of a pathogen."
    ]


def test_extract_claims_splits_only_when_both_conjoined_parts_are_statements():
    response = "Canberra is the capital of Australia and Sydney is the capital of New South Wales."
    claims = extract_claims(response)

    assert [claim.text for claim in claims] == [
        "Canberra is the capital of Australia",
        "Sydney is the capital of New South Wales.",
    ]
    assert [response[claim.start : claim.end] for claim in claims] == [
        claim.text for claim in claims
    ]


def test_extract_claims_keeps_complete_claim_with_coordinated_predicates():
    claims = extract_claims("Vaccines prevent infection or reduce severe disease.")

    assert [claim.text for claim in claims] == [
        "Vaccines prevent infection or reduce severe disease."
    ]


def test_extract_claims_handles_quoted_sentence_endings_and_numbered_lists():
    response = (
        'This is a normal reaction and is called a "reaction site."\n\n'
        "2. Some people experience fatigue."
    )

    claims = extract_claims(response)

    assert [claim.text for claim in claims] == [
        'This is a normal reaction and is called a "reaction site."',
        "Some people experience fatigue.",
    ]
    assert [response[claim.start : claim.end] for claim in claims] == [
        claim.text for claim in claims
    ]


def test_extract_claims_keeps_complete_numeric_claims():
    claims = extract_claims("The year is 2020.")

    assert [claim.text for claim in claims] == ["The year is 2020."]


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
import pytest

from hallucination_detector.domain import Claim, Evidence, VerificationLabel
from hallucination_detector.retrieval import RetrievedEvidence
from hallucination_detector.verification import (
    EmbeddingSimilarityScorer,
    HuggingFaceNLIModel,
    NLIProbabilities,
    verify_claim_with_nli,
)


class FakeNLIModel:
    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)
        self.received_pairs = ()

    def predict_many(self, pairs):
        self.received_pairs = tuple(pairs)
        return self.probabilities


class FakeSimilarityScorer:
    def __init__(self, scores):
        self.scores = tuple(scores)

    def score_many(self, claim, evidence_texts):
        return self.scores


def make_retrieved(evidence_id="e1", text="Canberra is Australia's capital.", score=0.95):
    return RetrievedEvidence(
        evidence=Evidence(evidence_id, text, "reference"),
        bm25_score=2.0,
        embedding_score=0.9,
        combined_score=score,
        rank=1,
    )


def test_verification_returns_supported_for_entailing_evidence():
    claim = Claim("c1", "Canberra is the capital of Australia.")
    hit = make_retrieved()
    nli = FakeNLIModel((NLIProbabilities(0.97, 0.01, 0.02),))

    result = verify_claim_with_nli(claim, (hit,), nli, FakeSimilarityScorer((0.96,)))

    assert result.label is VerificationLabel.SUPPORTED
    assert result.evidence == (hit.evidence,)
    assert result.support_score == pytest.approx(0.97 * 0.95 * 0.96)
    assert result.evidence_scores[0].evidence.id == "e1"
    assert nli.received_pairs == ((hit.evidence.text, claim.text),)


def test_verification_returns_contradicted_for_conflicting_evidence():
    hit = make_retrieved(text="Canberra is not the capital of Australia.")
    nli = FakeNLIModel((NLIProbabilities(0.02, 0.96, 0.02),))

    result = verify_claim_with_nli(
        Claim("c1", "Canberra is the capital of Australia."),
        (hit,),
        nli,
        FakeSimilarityScorer((0.98,)),
    )

    assert result.label is VerificationLabel.CONTRADICTED
    assert result.contradiction_score == pytest.approx(0.96 * 0.95 * 0.98)
    assert "e1" in result.explanation


def test_verification_keeps_weak_or_neutral_evidence_unknown():
    hit = make_retrieved(score=0.3)
    nli = FakeNLIModel((NLIProbabilities(0.02, 0.03, 0.95),))

    result = verify_claim_with_nli(
        Claim("c1", "An unverified claim."),
        (hit,),
        nli,
        FakeSimilarityScorer((0.8,)),
    )

    assert result.label is VerificationLabel.UNKNOWN
    assert result.neutral_score == pytest.approx(0.95 * 0.3 * 0.8)


def test_no_retrieved_evidence_returns_unknown_without_model_calls():
    nli = FakeNLIModel(())
    result = verify_claim_with_nli(
        Claim("c1", "Claim with no evidence."),
        (),
        nli,
        FakeSimilarityScorer(()),
    )

    assert result.label is VerificationLabel.UNKNOWN
    assert result.evidence == ()
    assert result.support_score == 0.0
    assert nli.received_pairs == ()


def test_verifier_validates_thresholds_and_component_output_lengths():
    claim = Claim("c1", "A claim")
    hit = make_retrieved()
    with pytest.raises(ValueError, match="support_threshold"):
        verify_claim_with_nli(
            claim,
            (hit,),
            FakeNLIModel((NLIProbabilities(0.9, 0.05, 0.05),)),
            FakeSimilarityScorer((0.9,)),
            support_threshold=1.5,
        )

    with pytest.raises(ValueError, match="one score per"):
        verify_claim_with_nli(
            claim,
            (hit,),
            FakeNLIModel((NLIProbabilities(0.9, 0.05, 0.05),)),
            FakeSimilarityScorer(()),
        )

    with pytest.raises(ValueError, match="one probability distribution"):
        verify_claim_with_nli(
            claim,
            (hit,),
            FakeNLIModel(()),
            FakeSimilarityScorer((0.9,)),
        )


def test_nli_probabilities_are_validated_and_model_labels_are_resolved():
    with pytest.raises(ValueError, match="sum to 1"):
        NLIProbabilities(0.5, 0.5, 0.5)

    assert HuggingFaceNLIModel._resolve_labels(
        {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
        "some/nli-model",
        None,
    ) == {"contradiction": 0, "neutral": 1, "entailment": 2}
    assert HuggingFaceNLIModel._resolve_labels(
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        "some/nli-model",
        {"LABEL_0": "contradiction", "LABEL_1": "neutral", "LABEL_2": "entailment"},
    ) == {"contradiction": 0, "neutral": 1, "entailment": 2}

    with pytest.raises(ValueError, match="label_mapping"):
        HuggingFaceNLIModel._resolve_labels(
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
            "some/nli-model",
            None,
        )


def test_embedding_similarity_scorer_returns_cosine_scores_in_unit_interval():
    class FakeEmbedder:
        def encode(self, texts):
            vectors = {
                "claim": (1.0, 0.0),
                "same meaning": (1.0, 0.0),
                "orthogonal": (0.0, 1.0),
            }
            return tuple(vectors[text] for text in texts)

    scorer = EmbeddingSimilarityScorer(FakeEmbedder())

    assert scorer.score_many("claim", ("same meaning", "orthogonal")) == pytest.approx((1.0, 0.5))
import pytest

from hallucination_detector.domain import Claim, Evidence
from hallucination_detector.retrieval import HybridEvidenceRetriever
from hallucination_detector.retrieval_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class FakeEmbedder:
    """Small deterministic vectors for retrieval unit tests."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return tuple(self.vectors[text] for text in texts)


def test_hybrid_retriever_returns_scores_and_ranked_evidence():
    evidence = (
        Evidence("e1", "Marie Curie received the Nobel Prize in Chemistry in 1911."),
        Evidence("e2", "The Pacific Ocean is the largest ocean on Earth."),
    )
    embedder = FakeEmbedder(
        {
            evidence[0].text: (1.0, 0.0),
            evidence[1].text: (0.0, 1.0),
            "Marie Curie won the Nobel Prize for Chemistry in 1911.": (0.99, 0.01),
        }
    )
    retriever = HybridEvidenceRetriever(evidence, embedder)

    results = retriever.retrieve(
        Claim("c1", "Marie Curie won the Nobel Prize for Chemistry in 1911."),
        top_k=2,
    )

    assert [result.evidence.id for result in results] == ["e1", "e2"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].bm25_score > 0
    assert results[0].embedding_score > results[1].embedding_score
    assert 0 <= results[0].combined_score <= 1


def test_dense_signal_can_retrieve_a_paraphrase_with_little_lexical_overlap():
    evidence = (
        Evidence("e1", "A whale is a marine mammal."),
        Evidence("e2", "Canberra is Australia's national capital."),
    )
    embedder = FakeEmbedder(
        {
            evidence[0].text: (1.0, 0.0),
            evidence[1].text: (0.0, 1.0),
            "The largest animal lives in the ocean.": (0.99, 0.01),
        }
    )
    retriever = HybridEvidenceRetriever(
        evidence,
        embedder,
        bm25_weight=0.2,
        embedding_weight=0.8,
    )

    results = retriever.retrieve(Claim("c1", "The largest animal lives in the ocean."))

    assert results[0].evidence.id == "e1"
    assert results[0].embedding_score > results[1].embedding_score


def test_bm25_only_mode_does_not_require_an_embedder():
    retriever = HybridEvidenceRetriever(
        (Evidence("e1", "Canberra is the capital of Australia."),),
        embedding_weight=0.0,
        bm25_weight=1.0,
    )

    results = retriever.retrieve(Claim("c1", "What is the capital of Australia?"))

    assert results[0].evidence.id == "e1"
    assert results[0].embedding_score == 0.0


def test_retriever_validates_weights_and_unique_evidence_ids():
    with pytest.raises(ValueError, match="sum to 1.0"):
        HybridEvidenceRetriever((), bm25_weight=0.4, embedding_weight=0.4)

    with pytest.raises(ValueError, match="unique"):
        HybridEvidenceRetriever(
            (Evidence("same", "first"), Evidence("same", "second")),
            embedding_weight=0.0,
            bm25_weight=1.0,
        )


def test_retrieve_rejects_non_positive_top_k_and_empty_corpus_is_safe():
    retriever = HybridEvidenceRetriever((), embedding_weight=0.0, bm25_weight=1.0)

    assert retriever.retrieve(Claim("c1", "anything")) == ()
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve(Claim("c1", "anything"), top_k=0)


def test_retrieval_metrics_calculate_recall_precision_and_reciprocal_rank():
    ranking = ("e3", "e1", "e2")
    relevant = {"e1", "e2"}

    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert precision_at_k(ranking, relevant, 2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert mean_reciprocal_rank((ranking, ("e2", "e3")), (relevant, {"e2"})) == 0.75


def test_retrieval_metrics_validate_k_and_matching_query_counts():
    with pytest.raises(ValueError, match="k must"):
        recall_at_k(("e1",), {"e1"}, 0)
    with pytest.raises(ValueError, match="one relevant-ID"):
        mean_reciprocal_rank((("e1",),), ())
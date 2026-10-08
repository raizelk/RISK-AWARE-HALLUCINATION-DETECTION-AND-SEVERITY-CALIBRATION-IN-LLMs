import pytest

from hallucination_detector.domain import (
    Evidence,
    ModelResponse,
    VerificationLabel,
)
from hallucination_detector.evaluation import (
    ModelComparisonEvaluator,
    paired_bootstrap_mean_difference,
)
from hallucination_detector.retrieval import RetrievedEvidence
from hallucination_detector.verification import NLIProbabilities


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, claim, top_k=5):
        self.calls.append((claim.text, top_k))
        return (
            RetrievedEvidence(
                evidence=Evidence("gold-evidence", f"Evidence for: {claim.text}"),
                bm25_score=2.0,
                embedding_score=0.95,
                combined_score=0.95,
                rank=1,
            ),
        )


class FakeNLI:
    def predict_many(self, pairs):
        scores = []
        for _, hypothesis in pairs:
            if hypothesis.startswith("Supported"):
                scores.append(NLIProbabilities(0.96, 0.02, 0.02))
            elif hypothesis.startswith("Contradicted"):
                scores.append(NLIProbabilities(0.02, 0.96, 0.02))
            else:
                scores.append(NLIProbabilities(0.02, 0.03, 0.95))
        return tuple(scores)


class FakeSimilarity:
    def score_many(self, claim, evidence_texts):
        return (1.0,) * len(evidence_texts)


def make_evaluator(retriever=None, *, repetitions=250):
    return ModelComparisonEvaluator(
        model_names=("model-a", "model-b"),
        dataset_item_ids=("item-1", "item-2"),
        retriever=retriever or FakeRetriever(),
        nli_model=FakeNLI(),
        similarity_scorer=FakeSimilarity(),
        top_k=3,
        bootstrap_repetitions=repetitions,
        random_seed=42,
    )


def complete_response_matrix():
    return (
        ModelResponse("model-a", "item-1", "Supported claim.", 0.9),
        ModelResponse("model-b", "item-1", "Contradicted claim.", 0.2),
        ModelResponse("model-a", "item-2", "Supported claim.", 0.85),
        ModelResponse("model-b", "item-2", "Unknown claim.", 0.8),
    )


def test_evaluator_runs_same_components_for_full_matrix_and_reports_metrics():
    retriever = FakeRetriever()
    evaluator = make_evaluator(retriever)
    responses = complete_response_matrix()
    gold_labels = {
        ("model-a", "item-1", "claim-1"): VerificationLabel.SUPPORTED,
        ("model-b", "item-1", "claim-1"): VerificationLabel.CONTRADICTED,
        ("model-a", "item-2", "claim-1"): VerificationLabel.SUPPORTED,
        ("model-b", "item-2", "claim-1"): VerificationLabel.UNKNOWN,
    }
    consistency = {
        (response.model_name, response.dataset_item_id): 0.9
        for response in responses
    }
    gold_evidence = {key: {"gold-evidence"} for key in gold_labels}

    report = evaluator.evaluate(
        responses,
        consistency_scores=consistency,
        gold_labels=gold_labels,
        gold_evidence_ids=gold_evidence,
    )

    assert [(item.response.dataset_item_id, item.response.model_name) for item in report.responses] == [
        ("item-1", "model-a"),
        ("item-1", "model-b"),
        ("item-2", "model-a"),
        ("item-2", "model-b"),
    ]
    assert len(retriever.calls) == 4
    assert all(top_k == 3 for _, top_k in retriever.calls)

    model_a, model_b = report.model_metrics
    assert model_a.supported_rate == 1.0
    assert model_a.hallucination_rate == 0.0
    assert model_a.average_trust_score is not None
    assert model_a.verification_accuracy == 1.0
    assert model_a.verification_macro_f1 == 1.0
    assert model_a.retrieval_recall_at_k == 1.0
    assert model_a.confidence_brier_score is not None
    assert model_a.confidence_ece is not None

    assert model_b.contradicted_count == 1
    assert model_b.unknown_count == 1
    assert model_b.hallucination_rate == 0.5
    assert model_b.retrieval_recall_at_k == 1.0
    assert model_b.verification_accuracy == 1.0

    paired = report.paired_comparisons[0]
    assert paired.paired_item_count == 2
    assert paired.supported_rate_difference.mean_difference == 1.0
    assert paired.model_b_error_only_items == 1
    assert paired.model_a_error_only_items == 0
    assert paired.mcnemar_exact_p_value == 1.0


def test_trust_score_is_not_reported_without_external_consistency_signal():
    report = make_evaluator().evaluate(complete_response_matrix())

    assert all(item.trust_score is None for item in report.responses)
    assert all(metrics.average_trust_score is None for metrics in report.model_metrics)


@pytest.mark.parametrize(
    "responses, message",
    [
        (complete_response_matrix()[:-1], "incomplete"),
        (
            complete_response_matrix() + (complete_response_matrix()[0],),
            "Duplicate response",
        ),
        (
            complete_response_matrix()
            + (ModelResponse("other-model", "item-1", "Supported claim."),),
            "Unexpected model",
        ),
    ],
)
def test_evaluator_rejects_incomplete_duplicate_and_unknown_responses(responses, message):
    with pytest.raises(ValueError, match=message):
        make_evaluator().evaluate(responses)


def test_evaluator_rejects_unknown_gold_keys_and_invalid_consistency():
    evaluator = make_evaluator()
    with pytest.raises(ValueError, match="Gold labels"):
        evaluator.evaluate(
            complete_response_matrix(),
            gold_labels={("model-a", "missing", "claim-1"): VerificationLabel.SUPPORTED},
        )
    with pytest.raises(ValueError, match="consistency score"):
        evaluator.evaluate(
            complete_response_matrix(),
            consistency_scores={("model-a", "item-1"): 1.5},
        )


def test_paired_bootstrap_is_reproducible_and_validates_input():
    first = paired_bootstrap_mean_difference(
        ((0.8, 0.6), (0.9, 0.7)), repetitions=100, seed=9
    )
    second = paired_bootstrap_mean_difference(
        ((0.8, 0.6), (0.9, 0.7)), repetitions=100, seed=9
    )

    assert first == second
    assert first.mean_difference == pytest.approx(0.2)
    assert first.lower == pytest.approx(0.2)
    assert first.upper == pytest.approx(0.2)
    with pytest.raises(ValueError, match="paired observation"):
        paired_bootstrap_mean_difference(())
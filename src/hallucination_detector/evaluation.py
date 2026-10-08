"""Fair, paired evaluation of multiple models on one common dataset.

Every supplied response is processed by the same claim extractor, retriever,
NLI model, similarity scorer, severity classifier, and trust scorer. Statistical
comparisons are paired by dataset item because each model answers the same set.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .baseline import extract_claims
from .domain import (
    Claim,
    ModelResponse,
    Severity,
    VerificationLabel,
    VerificationResult,
)
from .retrieval import RetrievedEvidence
from .severity import SeverityAssessment, SeverityClassifier
from .trust import TrustScore, TrustScorer
from .verification import NLIModel, SimilarityScorer, verify_claim_with_nli

ClaimKey = tuple[str, str, str]
ResponseKey = tuple[str, str]


class EvidenceRetriever(Protocol):
    """Retrieval interface accepted by the evaluation pipeline."""

    def retrieve(self, claim: Claim, top_k: int = 5) -> Sequence[RetrievedEvidence]:
        """Return ranked evidence for a claim."""


@dataclass(frozen=True)
class EvaluatedClaim:
    """All shared-pipeline outputs for one response claim."""

    verification: VerificationResult
    severity: SeverityAssessment
    retrieved_evidence: tuple[RetrievedEvidence, ...]


@dataclass(frozen=True)
class EvaluatedResponse:
    """Claim-level analysis and optional trust score for one model response."""

    response: ModelResponse
    claims: tuple[EvaluatedClaim, ...]
    trust_score: TrustScore | None


@dataclass(frozen=True)
class ClassMetrics:
    """Per-label precision, recall, and F1 against manually supplied labels."""

    label: VerificationLabel
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ModelMetrics:
    """Aggregated factual and risk metrics for one model."""

    model_name: str
    response_count: int
    dataset_item_count: int
    claim_count: int
    supported_count: int
    contradicted_count: int
    unknown_count: int
    supported_rate: float | None
    contradicted_rate: float | None
    unknown_rate: float | None
    hallucination_rate: float | None
    severity_weighted_error_rate: float | None
    severity_counts: tuple[tuple[Severity, int], ...]
    average_trust_score: float | None
    verification_accuracy: float | None
    verification_macro_f1: float | None
    verification_class_metrics: tuple[ClassMetrics, ...]
    retrieval_k: int
    retrieval_recall_at_k: float | None
    confidence_brier_score: float | None
    confidence_ece: float | None


@dataclass(frozen=True)
class BootstrapInterval:
    """Paired bootstrap estimate and percentile confidence interval."""

    mean_difference: float
    lower: float
    upper: float
    sample_size: int
    repetitions: int
    confidence_level: float


@dataclass(frozen=True)
class PairedModelComparison:
    """Paired model comparison on item-level factuality outcomes."""

    model_a: str
    model_b: str
    paired_item_count: int
    supported_rate_difference: BootstrapInterval | None
    mcnemar_exact_p_value: float | None
    model_a_error_only_items: int
    model_b_error_only_items: int


@dataclass(frozen=True)
class ModelComparisonReport:
    """Per-response results, model aggregates, and paired comparisons."""

    responses: tuple[EvaluatedResponse, ...]
    model_metrics: tuple[ModelMetrics, ...]
    paired_comparisons: tuple[PairedModelComparison, ...]
    dataset_item_ids: tuple[str, ...]

    def save_json(self, path: str | Path) -> None:
        """Write this complete evaluation report as versioned JSON atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        payload = {"schema_version": 1, **asdict(self)}
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()


class ModelComparisonEvaluator:
    """Run the same fact-checking pipeline and compare configured models.

    ``consistency_scores`` are optional, externally measured per
    ``(model_name, dataset_item_id)`` (for example from repeated/paraphrased
    prompts). Trust scores are omitted when no consistency score is provided,
    rather than silently treating consistency as perfect.

    Optional ``gold_labels`` and ``gold_evidence_ids`` are keyed by
    ``(model_name, dataset_item_id, claim_id)``. They enable verification
    classification and retrieval metrics; they are not required to calculate
    descriptive output-label distributions.
    """

    def __init__(
        self,
        model_names: Sequence[str],
        dataset_item_ids: Sequence[str],
        retriever: EvidenceRetriever,
        nli_model: NLIModel,
        similarity_scorer: SimilarityScorer,
        *,
        top_k: int = 5,
        claim_extractor: Callable[[str], Sequence[Claim]] = extract_claims,
        severity_classifier: SeverityClassifier | None = None,
        trust_scorer: TrustScorer | None = None,
        support_threshold: float = 0.55,
        contradiction_threshold: float = 0.55,
        bootstrap_repetitions: int = 2000,
        bootstrap_confidence_level: float = 0.95,
        random_seed: int = 0,
    ) -> None:
        self._model_names = tuple(model_names)
        self._dataset_item_ids = tuple(dataset_item_ids)
        if not self._model_names or any(not name.strip() for name in self._model_names):
            raise ValueError("At least one non-empty model name is required.")
        if len(set(self._model_names)) != len(self._model_names):
            raise ValueError("Model names must be unique.")
        if not self._dataset_item_ids or any(not item_id.strip() for item_id in self._dataset_item_ids):
            raise ValueError("At least one non-empty dataset item ID is required.")
        if len(set(self._dataset_item_ids)) != len(self._dataset_item_ids):
            raise ValueError("Dataset item IDs must be unique.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions must be at least 1.")
        if (
            not math.isfinite(bootstrap_confidence_level)
            or not 0.0 < bootstrap_confidence_level < 1.0
        ):
            raise ValueError("bootstrap_confidence_level must be between 0 and 1.")

        self._retriever = retriever
        self._nli_model = nli_model
        self._similarity_scorer = similarity_scorer
        self._top_k = top_k
        self._claim_extractor = claim_extractor
        self._severity_classifier = severity_classifier or SeverityClassifier()
        self._trust_scorer = trust_scorer or TrustScorer()
        self._support_threshold = support_threshold
        self._contradiction_threshold = contradiction_threshold
        self._bootstrap_repetitions = bootstrap_repetitions
        self._bootstrap_confidence_level = bootstrap_confidence_level
        self._random_seed = random_seed

    def evaluate(
        self,
        responses: Sequence[ModelResponse],
        *,
        consistency_scores: Mapping[ResponseKey, float] | None = None,
        gold_labels: Mapping[ClaimKey, VerificationLabel] | None = None,
        gold_evidence_ids: Mapping[ClaimKey, Collection[str]] | None = None,
    ) -> ModelComparisonReport:
        """Evaluate a complete model-by-item response matrix and compare it."""
        response_by_key: dict[ResponseKey, ModelResponse] = {}
        for response in responses:
            if response.model_name not in self._model_names:
                raise ValueError(f"Unexpected model name: {response.model_name!r}.")
            if response.dataset_item_id not in self._dataset_item_ids:
                raise ValueError(f"Unexpected dataset item ID: {response.dataset_item_id!r}.")
            if not response.text.strip():
                raise ValueError("Model responses must contain non-whitespace text.")
            if response.confidence is not None and (
                not math.isfinite(response.confidence) or not 0.0 <= response.confidence <= 1.0
            ):
                raise ValueError("Response confidence must be finite and in [0, 1].")
            key = (response.model_name, response.dataset_item_id)
            if key in response_by_key:
                raise ValueError(f"Duplicate response for model/item pair {key!r}.")
            response_by_key[key] = response

        expected_keys = {
            (model_name, item_id)
            for item_id in self._dataset_item_ids
            for model_name in self._model_names
        }
        missing = expected_keys - response_by_key.keys()
        if missing:
            raise ValueError(f"Response matrix is incomplete; missing pairs: {sorted(missing)!r}.")

        consistency = dict(consistency_scores or {})
        if not consistency.keys() <= expected_keys:
            raise ValueError("Consistency scores contain unknown model/item pairs.")
        for key, score in consistency.items():
            _validate_unit_interval(score, f"consistency score for {key!r}")

        evaluated: list[EvaluatedResponse] = []
        for item_id in self._dataset_item_ids:
            for model_name in self._model_names:
                response = response_by_key[(model_name, item_id)]
                claims = tuple(self._claim_extractor(response.text))
                claim_ids = [claim.id for claim in claims]
                if len(set(claim_ids)) != len(claim_ids):
                    raise ValueError(
                        f"Claim extractor returned duplicate IDs for {model_name!r}/{item_id!r}."
                    )

                analyzed_claims: list[EvaluatedClaim] = []
                for claim in claims:
                    retrieved = tuple(self._retriever.retrieve(claim, top_k=self._top_k))
                    verification = verify_claim_with_nli(
                        claim,
                        retrieved,
                        self._nli_model,
                        self._similarity_scorer,
                        support_threshold=self._support_threshold,
                        contradiction_threshold=self._contradiction_threshold,
                    )
                    severity = self._severity_classifier.classify(verification)
                    analyzed_claims.append(
                        EvaluatedClaim(
                            verification=verification,
                            severity=severity,
                            retrieved_evidence=retrieved,
                        )
                    )

                claim_verifications = tuple(item.verification for item in analyzed_claims)
                trust_score = None
                response_key = (model_name, item_id)
                if claim_verifications and response_key in consistency:
                    confidences = (response.confidence,) * len(claim_verifications)
                    trust_score = self._trust_scorer.score(
                        claim_verifications,
                        severity_assessments=tuple(item.severity for item in analyzed_claims),
                        consistency_score=consistency[response_key],
                        confidences=confidences,
                    )

                evaluated.append(
                    EvaluatedResponse(
                        response=response,
                        claims=tuple(analyzed_claims),
                        trust_score=trust_score,
                    )
                )

        evaluated_responses = tuple(evaluated)
        all_claim_keys = {
            (item.response.model_name, item.response.dataset_item_id, claim.verification.claim.id)
            for item in evaluated_responses
            for claim in item.claims
        }
        gold_label_map = dict(gold_labels or {})
        gold_evidence_map = dict(gold_evidence_ids or {})
        if not gold_label_map.keys() <= all_claim_keys:
            raise ValueError("Gold labels contain keys that do not match evaluated claims.")
        if not gold_evidence_map.keys() <= all_claim_keys:
            raise ValueError("Gold evidence IDs contain keys that do not match evaluated claims.")

        model_metrics = tuple(
            self._aggregate_model(
                model_name,
                evaluated_responses,
                gold_label_map,
                gold_evidence_map,
            )
            for model_name in self._model_names
        )
        comparisons = tuple(
            self._compare_models(model_a, model_b, evaluated_responses)
            for model_a, model_b in itertools.combinations(self._model_names, 2)
        )
        return ModelComparisonReport(
            responses=evaluated_responses,
            model_metrics=model_metrics,
            paired_comparisons=comparisons,
            dataset_item_ids=self._dataset_item_ids,
        )

    def _aggregate_model(
        self,
        model_name: str,
        responses: tuple[EvaluatedResponse, ...],
        gold_labels: Mapping[ClaimKey, VerificationLabel],
        gold_evidence_ids: Mapping[ClaimKey, Collection[str]],
    ) -> ModelMetrics:
        model_responses = tuple(item for item in responses if item.response.model_name == model_name)
        claims = tuple((response, claim) for response in model_responses for claim in response.claims)
        count = len(claims)
        label_counts = Counter(item.verification.label for _, item in claims)
        supported = label_counts[VerificationLabel.SUPPORTED]
        contradicted = label_counts[VerificationLabel.CONTRADICTED]
        unknown = label_counts[VerificationLabel.UNKNOWN]

        severity_values = tuple(
            item.severity.severity
            for _, item in claims
            if item.severity.severity is not None
        )
        severity_counts = Counter(severity_values)
        severity_weights = {
            Severity.MILD: 1,
            Severity.MODERATE: 2,
            Severity.SEVERE: 4,
            Severity.CRITICAL: 8,
        }
        weighted_error_rate = (
            sum(severity_weights[severity] for severity in severity_values) / (8 * count)
            if count
            else None
        )

        predicted_gold_pairs = []
        for response, claim_result in claims:
            claim = claim_result.verification.claim
            key = (model_name, response.response.dataset_item_id, claim.id)
            if key in gold_labels:
                predicted_gold_pairs.append((claim_result.verification.label, gold_labels[key]))

        class_metrics = _classification_metrics(predicted_gold_pairs)
        verification_accuracy = (
            sum(predicted == gold for predicted, gold in predicted_gold_pairs)
            / len(predicted_gold_pairs)
            if predicted_gold_pairs
            else None
        )
        macro_f1 = (
            sum(metric.f1 for metric in class_metrics) / len(class_metrics)
            if class_metrics
            else None
        )

        recall_values: list[float] = []
        for response, claim_result in claims:
            claim = claim_result.verification.claim
            key = (model_name, response.response.dataset_item_id, claim.id)
            if key in gold_evidence_ids:
                relevant = set(gold_evidence_ids[key])
                if relevant:
                    retrieved_ids = {hit.evidence.id for hit in claim_result.retrieved_evidence}
                    recall_values.append(len(retrieved_ids & relevant) / len(relevant))

        trust_values = tuple(
            item.trust_score.score for item in model_responses if item.trust_score is not None
        )
        brier, ece = _response_confidence_calibration(model_responses, gold_labels)

        def rate(value: int) -> float | None:
            return value / count if count else None

        return ModelMetrics(
            model_name=model_name,
            response_count=len(model_responses),
            dataset_item_count=len({item.response.dataset_item_id for item in model_responses}),
            claim_count=count,
            supported_count=supported,
            contradicted_count=contradicted,
            unknown_count=unknown,
            supported_rate=rate(supported),
            contradicted_rate=rate(contradicted),
            unknown_rate=rate(unknown),
            hallucination_rate=rate(contradicted),
            severity_weighted_error_rate=weighted_error_rate,
            severity_counts=tuple((severity, severity_counts[severity]) for severity in Severity),
            average_trust_score=sum(trust_values) / len(trust_values) if trust_values else None,
            verification_accuracy=verification_accuracy,
            verification_macro_f1=macro_f1,
            verification_class_metrics=class_metrics,
            retrieval_k=self._top_k,
            retrieval_recall_at_k=sum(recall_values) / len(recall_values) if recall_values else None,
            confidence_brier_score=brier,
            confidence_ece=ece,
        )

    def _compare_models(
        self,
        model_a: str,
        model_b: str,
        responses: tuple[EvaluatedResponse, ...],
    ) -> PairedModelComparison:
        by_pair = {
            (item.response.model_name, item.response.dataset_item_id): item
            for item in responses
        }
        paired_rates: list[tuple[float, float]] = []
        a_only = 0
        b_only = 0
        for item_id in self._dataset_item_ids:
            result_a = by_pair[(model_a, item_id)]
            result_b = by_pair[(model_b, item_id)]
            labels_a = tuple(claim.verification.label for claim in result_a.claims)
            labels_b = tuple(claim.verification.label for claim in result_b.claims)
            if labels_a and labels_b:
                paired_rates.append(
                    (
                        sum(label is VerificationLabel.SUPPORTED for label in labels_a) / len(labels_a),
                        sum(label is VerificationLabel.SUPPORTED for label in labels_b) / len(labels_b),
                    )
                )
            error_a = any(label is VerificationLabel.CONTRADICTED for label in labels_a)
            error_b = any(label is VerificationLabel.CONTRADICTED for label in labels_b)
            if error_a and not error_b:
                a_only += 1
            elif error_b and not error_a:
                b_only += 1

        interval = (
            paired_bootstrap_mean_difference(
                paired_rates,
                repetitions=self._bootstrap_repetitions,
                confidence_level=self._bootstrap_confidence_level,
                seed=self._random_seed,
            )
            if paired_rates
            else None
        )
        return PairedModelComparison(
            model_a=model_a,
            model_b=model_b,
            paired_item_count=len(paired_rates),
            supported_rate_difference=interval,
            mcnemar_exact_p_value=_exact_mcnemar_p_value(a_only, b_only),
            model_a_error_only_items=a_only,
            model_b_error_only_items=b_only,
        )


def paired_bootstrap_mean_difference(
    paired_values: Sequence[tuple[float, float]],
    *,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap paired mean(A-B) with deterministic percentile bounds."""
    if not paired_values:
        raise ValueError("At least one paired observation is required.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    for left, right in paired_values:
        _validate_unit_interval(left, "left paired value")
        _validate_unit_interval(right, "right paired value")

    differences = [left - right for left, right in paired_values]
    observed = sum(differences) / len(differences)
    generator = random.Random(seed)
    bootstrap_means = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(repetitions)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        mean_difference=observed,
        lower=_percentile(bootstrap_means, alpha),
        upper=_percentile(bootstrap_means, 1.0 - alpha),
        sample_size=len(differences),
        repetitions=repetitions,
        confidence_level=confidence_level,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _exact_mcnemar_p_value(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant paired items."""
    if a_only < 0 or b_only < 0:
        raise ValueError("Discordant counts cannot be negative.")
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _classification_metrics(
    pairs: Sequence[tuple[VerificationLabel, VerificationLabel]],
) -> tuple[ClassMetrics, ...]:
    if not pairs:
        return ()
    metrics: list[ClassMetrics] = []
    for label in VerificationLabel:
        true_positive = sum(predicted is label and gold is label for predicted, gold in pairs)
        false_positive = sum(predicted is label and gold is not label for predicted, gold in pairs)
        false_negative = sum(predicted is not label and gold is label for predicted, gold in pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(gold is label for _, gold in pairs)
        metrics.append(ClassMetrics(label, precision, recall, f1, support))
    return tuple(metrics)


def _response_confidence_calibration(
    responses: Sequence[EvaluatedResponse],
    gold_labels: Mapping[ClaimKey, VerificationLabel],
    *,
    bucket_count: int = 10,
) -> tuple[float | None, float | None]:
    """Measure response confidence against fully gold-labeled response correctness.

    A response is considered correct only when every extracted claim has a gold
    label and every prediction matches it; this avoids treating UNKNOWN as
    false without ground truth. Missing/incomplete labels make calibration
    unavailable for that response.
    """
    confidence_outcomes: list[tuple[float, float]] = []
    for response in responses:
        confidence = response.response.confidence
        if confidence is None or not response.claims:
            continue
        gold_values = []
        complete = True
        for claim_result in response.claims:
            claim = claim_result.verification.claim
            key = (response.response.model_name, response.response.dataset_item_id, claim.id)
            if key not in gold_labels:
                complete = False
                break
            gold_values.append(
                float(claim_result.verification.label is gold_labels[key])
            )
        if complete:
            correctness = float(all(value == 1.0 for value in gold_values))
            confidence_outcomes.append((confidence, correctness))

    if not confidence_outcomes:
        return None, None
    brier = sum((confidence - outcome) ** 2 for confidence, outcome in confidence_outcomes) / len(confidence_outcomes)
    calibration_error = 0.0
    for bucket in range(bucket_count):
        lower = bucket / bucket_count
        upper = (bucket + 1) / bucket_count
        values = [
            pair for pair in confidence_outcomes
            if lower <= pair[0] < upper or (bucket == bucket_count - 1 and pair[0] == 1.0)
        ]
        if values:
            mean_confidence = sum(confidence for confidence, _ in values) / len(values)
            mean_accuracy = sum(outcome for _, outcome in values) / len(values)
            calibration_error += len(values) / len(confidence_outcomes) * abs(mean_confidence - mean_accuracy)
    return brier, calibration_error


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
"""Development-set threshold tuning and ordinal severity evaluation."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .annotations import ClaimKey
from .domain import Claim, Severity, VerificationLabel, VerificationResult
from .severity import SeverityClassifier
from .verification import classify_verification_scores


@dataclass(frozen=True)
class CalibrationObservation:
    key: ClaimKey
    support_score: float
    contradiction_score: float
    gold_label: VerificationLabel


@dataclass(frozen=True)
class ThresholdCalibration:
    support_threshold: float
    contradiction_threshold: float
    objective: str
    objective_score: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    coverage: float
    sample_size: int
    candidates_evaluated: int


@dataclass(frozen=True)
class SeverityMetrics:
    accuracy: float
    macro_f1: float
    quadratic_weighted_kappa: float | None
    sample_size: int
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SeverityCalibrationObservation:
    key: ClaimKey
    claim: Claim
    verification_label: VerificationLabel
    contradiction_score: float
    gold_severity: Severity | None


@dataclass(frozen=True)
class SeverityThresholdCalibration:
    high_contradiction_threshold: float
    metrics: SeverityMetrics
    candidates_evaluated: int


_SEVERITY_ORDER = tuple(Severity)


def calibrate_verification_thresholds(
    observations: Sequence[CalibrationObservation],
    *,
    candidate_thresholds: Sequence[float] | None = None,
    objective: str = "macro_f1",
) -> ThresholdCalibration:
    """Grid-search support and contradiction cutoffs using development labels only.

    Scores are the cached maximum adjusted NLI scores from an evaluation report.
    The tie-breaker prefers higher accuracy, then coverage, then values closest
    to the provisional 0.55 defaults.
    """
    if not observations:
        raise ValueError("At least one development observation is required.")
    if objective not in {"macro_f1", "balanced_accuracy"}:
        raise ValueError("objective must be 'macro_f1' or 'balanced_accuracy'.")
    keys = [observation.key for observation in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("Calibration observations must have unique claim keys.")
    for observation in observations:
        _validate_probability("support_score", observation.support_score)
        _validate_probability("contradiction_score", observation.contradiction_score)
        if not isinstance(observation.gold_label, VerificationLabel):
            raise ValueError("gold_label must be a VerificationLabel.")

    candidates = (
        tuple(index / 20 for index in range(21))
        if candidate_thresholds is None
        else tuple(candidate_thresholds)
    )
    if not candidates:
        raise ValueError("At least one threshold candidate is required.")
    candidates = tuple(sorted(set(candidates)))
    for threshold in candidates:
        _validate_probability("candidate threshold", threshold)

    best: ThresholdCalibration | None = None
    best_rank: tuple[float, float, float, float] | None = None
    for support_threshold in candidates:
        for contradiction_threshold in candidates:
            predictions = tuple(
                classify_verification_scores(
                    item.support_score,
                    item.contradiction_score,
                    support_threshold=support_threshold,
                    contradiction_threshold=contradiction_threshold,
                )
                for item in observations
            )
            gold = tuple(item.gold_label for item in observations)
            accuracy, macro_f1, balanced_accuracy = _classification_scores(gold, predictions)
            coverage = sum(label is not VerificationLabel.UNKNOWN for label in predictions) / len(predictions)
            objective_score = macro_f1 if objective == "macro_f1" else balanced_accuracy
            default_distance = abs(support_threshold - 0.55) + abs(contradiction_threshold - 0.55)
            rank = (objective_score, accuracy, coverage, -default_distance)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = ThresholdCalibration(
                    support_threshold=support_threshold,
                    contradiction_threshold=contradiction_threshold,
                    objective=objective,
                    objective_score=objective_score,
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                    balanced_accuracy=balanced_accuracy,
                    coverage=coverage,
                    sample_size=len(observations),
                    candidates_evaluated=len(candidates) ** 2,
                )
    assert best is not None
    return best


def calibrate_severity_threshold(
    observations: Sequence[SeverityCalibrationObservation],
    *,
    candidate_thresholds: Sequence[float] | None = None,
) -> SeverityThresholdCalibration:
    """Tune the rule-based contradiction escalation cutoff against dev severity labels."""
    if not observations:
        raise ValueError("At least one development severity observation is required.")
    keys = [observation.key for observation in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("Severity observations must have unique claim keys.")
    for observation in observations:
        _validate_probability("contradiction_score", observation.contradiction_score)
        if not isinstance(observation.verification_label, VerificationLabel):
            raise ValueError("verification_label must be a VerificationLabel.")

    candidates = (
        tuple(index / 20 for index in range(21))
        if candidate_thresholds is None
        else tuple(candidate_thresholds)
    )
    if not candidates:
        raise ValueError("At least one severity threshold candidate is required.")
    candidates = tuple(sorted(set(candidates)))
    for threshold in candidates:
        _validate_probability("candidate threshold", threshold)

    gold = {item.key: item.gold_severity for item in observations}
    best: SeverityThresholdCalibration | None = None
    best_rank: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        classifier = SeverityClassifier(high_contradiction_threshold=threshold)
        predicted: dict[ClaimKey, Severity | None] = {}
        for item in observations:
            verification = VerificationResult(
                claim=item.claim,
                label=item.verification_label,
                contradiction_score=item.contradiction_score,
            )
            predicted[item.key] = classifier.classify(verification).severity
        metrics = evaluate_severity(gold, predicted)
        kappa = metrics.quadratic_weighted_kappa
        rank = (
            metrics.macro_f1,
            metrics.accuracy,
            kappa if kappa is not None else -1.0,
            -abs(threshold - 0.8),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = SeverityThresholdCalibration(threshold, metrics, len(candidates))
    assert best is not None
    return best


def evaluate_severity(
    gold: Mapping[ClaimKey, Severity | None],
    predicted: Mapping[ClaimKey, Severity | None],
) -> SeverityMetrics:
    """Compare non-null gold severity labels with predictions using ordinal metrics."""
    gold_keys = {key for key, severity in gold.items() if severity is not None}
    if not gold_keys:
        raise ValueError("At least one non-null gold severity label is required.")
    missing = gold_keys - predicted.keys()
    if missing:
        raise ValueError(f"Missing predicted severity for claim keys: {sorted(missing)!r}.")

    gold_values: list[Severity] = []
    predicted_values: list[Severity] = []
    for key in sorted(gold_keys):
        gold_severity = gold[key]
        predicted_severity = predicted[key]
        if predicted_severity is None:
            raise ValueError(f"Predicted severity is null for labeled claim {key!r}.")
        try:
            gold_values.append(Severity(gold_severity))
            predicted_values.append(Severity(predicted_severity))
        except ValueError as error:
            raise ValueError(f"Invalid severity value for claim {key!r}.") from error

    labels = _SEVERITY_ORDER
    confusion_counts = Counter(zip(gold_values, predicted_values))
    matrix = tuple(
        tuple(confusion_counts[(gold_label, predicted_label)] for predicted_label in labels)
        for gold_label in labels
    )
    accuracy = sum(gold_label == predicted_label for gold_label, predicted_label in zip(gold_values, predicted_values)) / len(gold_values)
    active_labels = set(gold_values) | set(predicted_values)
    macro_f1 = _macro_f1(gold_values, predicted_values, active_labels)
    weighted_kappa = _quadratic_weighted_kappa(matrix)
    return SeverityMetrics(accuracy, macro_f1, weighted_kappa, len(gold_values), matrix)


def _classification_scores(
    gold: Sequence[VerificationLabel],
    predicted: Sequence[VerificationLabel],
) -> tuple[float, float, float]:
    accuracy = sum(left == right for left, right in zip(gold, predicted)) / len(gold)
    active_labels = set(gold)
    macro_f1 = _macro_f1(gold, predicted, active_labels)
    recalls = [
        sum(actual == label and guess == label for actual, guess in zip(gold, predicted))
        / sum(actual == label for actual in gold)
        for label in active_labels
    ]
    return accuracy, macro_f1, sum(recalls) / len(recalls)


def _macro_f1(gold: Sequence[object], predicted: Sequence[object], labels: set[object]) -> float:
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(gold, predicted))
        false_positive = sum(actual != label and guess == label for actual, guess in zip(gold, predicted))
        false_negative = sum(actual == label and guess != label for actual, guess in zip(gold, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(f1_values) / len(f1_values)


def _quadratic_weighted_kappa(matrix: tuple[tuple[int, ...], ...]) -> float | None:
    sample_size = sum(sum(row) for row in matrix)
    if sample_size < 1:
        return None
    category_count = len(matrix)
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(category_count)) for column in range(category_count)]
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for row in range(category_count):
        for column in range(category_count):
            weight = ((row - column) / max(category_count - 1, 1)) ** 2
            observed_disagreement += weight * matrix[row][column] / sample_size
            expected_count = row_totals[row] * column_totals[column] / sample_size
            expected_disagreement += weight * expected_count / sample_size
    if expected_disagreement == 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")


def create_topic_split(
    topic_by_item: Mapping[str, str],
    *,
    development_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Create deterministic train/dev/test assignments with topics kept intact."""
    if not topic_by_item:
        raise ValueError("At least one dataset item is required for a topic split.")
    for name, fraction in (
        ("development_fraction", development_fraction),
        ("test_fraction", test_fraction),
    ):
        _validate_probability(name, fraction)
        if fraction == 0.0:
            raise ValueError(f"{name} must be greater than zero.")
    if development_fraction + test_fraction >= 1.0:
        raise ValueError("Development and test fractions must sum to less than 1.")
    if any(
        not isinstance(item_id, str)
        or not item_id.strip()
        or not isinstance(topic, str)
        or not topic.strip()
        for item_id, topic in topic_by_item.items()
    ):
        raise ValueError("Dataset item IDs and topics must be non-empty strings.")

    topics = sorted(set(topic_by_item.values()))
    if len(topics) < 3:
        raise ValueError("At least three topics are required for train/dev/test separation.")
    development_count = max(1, round(len(topics) * development_fraction))
    test_count = max(1, round(len(topics) * test_fraction))
    if development_count + test_count >= len(topics):
        raise ValueError("Fractions leave no topic for the training split.")

    random.Random(seed).shuffle(topics)
    test_topics = set(topics[:test_count])
    development_topics = set(topics[test_count:test_count + development_count])
    return {
        item_id: (
            "test" if topic in test_topics
            else "dev" if topic in development_topics
            else "train"
        )
        for item_id, topic in topic_by_item.items()
    }
"""Semantic and NLI-based verification of claims against retrieved evidence.

Stage 5 consumes Stage 4 ``RetrievedEvidence`` candidates. Provider/model
adapters are injectable so the decision logic remains testable and replaceable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .domain import (
    Claim,
    VerificationEvidenceScore,
    VerificationLabel,
    VerificationResult,
)
from .retrieval import RetrievedEvidence, TextEmbedder


@dataclass(frozen=True)
class NLIProbabilities:
    """Normalized entailment, contradiction, and neutral probabilities."""

    entailment: float
    contradiction: float
    neutral: float

    def __post_init__(self) -> None:
        values = (self.entailment, self.contradiction, self.neutral)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("NLI probabilities must be finite values in [0, 1].")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-3):
            raise ValueError("Entailment, contradiction, and neutral probabilities must sum to 1.")


class NLIModel(Protocol):
    """Batch premise/hypothesis pairs and return one NLI distribution per pair."""

    def predict_many(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[NLIProbabilities]:
        """Return NLI probabilities in the same order as input pairs."""


class SimilarityScorer(Protocol):
    """Score semantic relatedness between one claim and evidence passages."""

    def score_many(self, claim: str, evidence_texts: Sequence[str]) -> Sequence[float]:
        """Return similarity scores in [0, 1], preserving input order."""


class EmbeddingSimilarityScorer:
    """Cosine similarity scorer using the Stage 4 text-embedding interface."""

    def __init__(self, embedder: TextEmbedder) -> None:
        self._embedder = embedder

    def score_many(self, claim: str, evidence_texts: Sequence[str]) -> Sequence[float]:
        if not evidence_texts:
            return ()
        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in self._embedder.encode((claim, *evidence_texts))
        )
        if len(vectors) != len(evidence_texts) + 1:
            raise ValueError("Embedder must return one vector per input text.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("Embedding vectors must have the same non-zero dimension.")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embedding vectors must contain only finite values.")

        query_vector = vectors[0]
        return tuple(
            self._cosine_similarity(query_vector, evidence_vector)
            for evidence_vector in vectors[1:]
        )

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        # Match Stage 4's documented range mapping from cosine [-1, 1] to [0, 1].
        return min(1.0, max(0.0, (cosine + 1.0) / 2.0))


class HuggingFaceNLIModel:
    """Optional transformer NLI model adapter with explicit label validation.

    ``facebook/bart-large-mnli`` is the default. Other models are accepted when
    their config labels clearly identify entailment, contradiction, and neutral;
    ambiguous ``LABEL_n`` configurations require an explicit label mapping.
    """

    _DEFAULT_MODEL = "facebook/bart-large-mnli"
    _DEFAULT_LABEL_IDS = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str | None = None,
        label_mapping: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Hugging Face NLI requires the optional verification extra. "
                "Install with `pip install -e .[verification]`."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._label_indices = self._resolve_labels(
            self._model.config.id2label,
            model_name,
            label_mapping,
        )

    @classmethod
    def _resolve_labels(
        cls,
        id_to_label: Mapping[int | str, str],
        model_name: str,
        label_mapping: Mapping[str, str] | None,
    ) -> dict[str, int]:
        normalized_mapping = (
            {key.strip().lower(): value.strip().lower() for key, value in label_mapping.items()}
            if label_mapping is not None
            else {}
        )
        resolved: dict[str, int] = {}

        for raw_index, raw_label in id_to_label.items():
            index = int(raw_index)
            label = raw_label.strip().lower()
            category = normalized_mapping.get(label)
            if category is None:
                if "contradiction" in label:
                    category = "contradiction"
                elif "entailment" in label and "not_entailment" not in label:
                    category = "entailment"
                elif "neutral" in label:
                    category = "neutral"
            if category in {"entailment", "contradiction", "neutral"}:
                if category in resolved:
                    raise ValueError(f"NLI model config has multiple labels for {category!r}.")
                resolved[category] = index

        if not resolved and model_name == cls._DEFAULT_MODEL:
            return {category: index for index, category in cls._DEFAULT_LABEL_IDS.items()}
        if set(resolved) != {"entailment", "contradiction", "neutral"}:
            raise ValueError(
                "Could not map model labels to entailment/contradiction/neutral. "
                "Provide label_mapping={model_label: category, ...}."
            )
        return resolved

    def predict_many(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[NLIProbabilities]:
        if not pairs:
            return ()
        premises = [premise for premise, _ in pairs]
        hypotheses = [hypothesis for _, hypothesis in pairs]
        encoded = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()

        expected_indices = max(self._label_indices.values())
        if any(len(row) <= expected_indices for row in probabilities):
            raise ValueError("NLI model output does not contain all mapped label logits.")
        return tuple(
            NLIProbabilities(
                entailment=float(row[self._label_indices["entailment"]]),
                contradiction=float(row[self._label_indices["contradiction"]]),
                neutral=float(row[self._label_indices["neutral"]]),
            )
            for row in probabilities
        )


def verify_claim_with_nli(
    claim: Claim,
    retrieved: Sequence[RetrievedEvidence],
    nli_model: NLIModel,
    similarity_scorer: SimilarityScorer,
    *,
    support_threshold: float = 0.55,
    contradiction_threshold: float = 0.55,
) -> VerificationResult:
    """Aggregate semantic similarity and NLI signals for a claim.

    For each candidate, adjusted support and contradiction are respectively
    ``NLI probability × retrieval score × semantic similarity``. The strongest
    adjusted evidence determines the label if it clears its threshold and
    strictly exceeds the competing signal; otherwise the claim remains UNKNOWN.
    Thresholds are initial operating points and should be calibrated on a
    labeled validation set before research conclusions are drawn.
    """
    _validate_threshold("support_threshold", support_threshold)
    _validate_threshold("contradiction_threshold", contradiction_threshold)

    candidates = tuple(retrieved)
    if not candidates:
        return VerificationResult(
            claim=claim,
            label=VerificationLabel.UNKNOWN,
            explanation="No evidence passages were retrieved for this claim.",
        )

    evidence_texts = tuple(hit.evidence.text for hit in candidates)
    similarities = tuple(float(score) for score in similarity_scorer.score_many(claim.text, evidence_texts))
    if len(similarities) != len(candidates):
        raise ValueError("Similarity scorer must return one score per evidence passage.")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in similarities):
        raise ValueError("Similarity scores must be finite values in [0, 1].")

    nli_scores = tuple(
        nli_model.predict_many(tuple((hit.evidence.text, claim.text) for hit in candidates))
    )
    if len(nli_scores) != len(candidates):
        raise ValueError("NLI model must return one probability distribution per evidence passage.")

    evidence_scores: list[VerificationEvidenceScore] = []
    for hit, similarity, probabilities in zip(candidates, similarities, nli_scores):
        retrieval_score = float(hit.combined_score)
        if not math.isfinite(retrieval_score) or not 0.0 <= retrieval_score <= 1.0:
            raise ValueError("Retrieved combined scores must be finite values in [0, 1].")
        evidence_scores.append(
            VerificationEvidenceScore(
                evidence=hit.evidence,
                retrieval_score=retrieval_score,
                similarity=similarity,
                entailment_probability=probabilities.entailment,
                contradiction_probability=probabilities.contradiction,
                neutral_probability=probabilities.neutral,
                support_score=probabilities.entailment * retrieval_score * similarity,
                contradiction_score=probabilities.contradiction * retrieval_score * similarity,
            )
        )

    best_support = max((item.support_score for item in evidence_scores), default=0.0)
    best_contradiction = max((item.contradiction_score for item in evidence_scores), default=0.0)
    best_neutral = max(
        (
            item.neutral_probability * item.retrieval_score * item.similarity
            for item in evidence_scores
        ),
        default=0.0,
    )
    best_similarity = max(similarities, default=0.0)

    support_evidence = max(evidence_scores, key=lambda item: item.support_score)
    contradiction_evidence = max(evidence_scores, key=lambda item: item.contradiction_score)

    if best_support >= support_threshold and best_support > best_contradiction:
        label = VerificationLabel.SUPPORTED
        explanation = (
            f"Evidence {support_evidence.evidence.id!r} most strongly entails the claim "
            f"(adjusted support score {best_support:.3f})."
        )
    elif best_contradiction >= contradiction_threshold and best_contradiction > best_support:
        label = VerificationLabel.CONTRADICTED
        explanation = (
            f"Evidence {contradiction_evidence.evidence.id!r} most strongly contradicts "
            f"the claim (adjusted contradiction score {best_contradiction:.3f})."
        )
    else:
        label = VerificationLabel.UNKNOWN
        explanation = (
            "Retrieved evidence did not produce a decisive entailment or contradiction "
            f"score (support {best_support:.3f}; contradiction {best_contradiction:.3f})."
        )

    return VerificationResult(
        claim=claim,
        label=label,
        evidence=tuple(hit.evidence for hit in candidates),
        similarity=best_similarity,
        explanation=explanation,
        support_score=best_support,
        contradiction_score=best_contradiction,
        neutral_score=best_neutral,
        evidence_scores=tuple(evidence_scores),
    )


def _validate_threshold(name: str, threshold: float) -> None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1].")
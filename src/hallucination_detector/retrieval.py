"""Hybrid BM25 and dense-vector retrieval for evidence passages.

The BM25 implementation is dependency-free. Dense retrieval uses an injected
embedder, with an optional Sentence Transformers adapter for local embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from .domain import Claim, Evidence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


class TextEmbedder(Protocol):
    """Encode a batch of texts into same-dimensional numeric vectors."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector for each input text."""


class SentenceTransformerEmbedder:
    """Optional dense embedder backed by the ``sentence-transformers`` package.

    The model is loaded when this class is constructed. The first run may need
    network access to download the model; later runs normally use its cache.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Dense retrieval requires the optional dependency. Install with "
                "`pip install -e .[retrieval]`."
            ) from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in row) for row in vectors)


@dataclass(frozen=True)
class RetrievedEvidence:
    """A passage and the retrieval signals used to rank it."""

    evidence: Evidence
    bm25_score: float
    embedding_score: float
    combined_score: float
    rank: int


class HybridEvidenceRetriever:
    """Retrieve top evidence with BM25 and optional dense embeddings.

    BM25 scores are normalized against the highest BM25 score for the current
    query. Cosine similarity is mapped from [-1, 1] to [0, 1]. The weighted
    combination is therefore interpretable and bounded in [0, 1].

    Pass an embedder to enable dense retrieval. Set ``embedding_weight=0`` to
    use BM25 by itself without installing the optional ML dependency.
    """

    def __init__(
        self,
        evidence: Sequence[Evidence],
        embedder: TextEmbedder | None = None,
        *,
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(bm25_weight) or not math.isfinite(embedding_weight):
            raise ValueError("Retrieval weights must be finite numbers.")
        if bm25_weight < 0 or embedding_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if not math.isclose(bm25_weight + embedding_weight, 1.0, abs_tol=1e-9):
            raise ValueError("BM25 and embedding weights must sum to 1.0.")
        if embedding_weight > 0 and embedder is None:
            raise ValueError("An embedder is required when embedding_weight is greater than 0.")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a positive finite number.")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self._evidence = tuple(evidence)
        ids = [item.id for item in self._evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence IDs must be unique within a retrieval index.")

        self._embedder = embedder
        self._bm25_weight = bm25_weight
        self._embedding_weight = embedding_weight
        self._k1 = k1
        self._b = b
        self._tokens = tuple(self._tokenize(item.text) for item in self._evidence)
        self._document_lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequency = Counter(
            token for document in self._tokens for token in set(document)
        )
        self._document_embeddings: tuple[tuple[float, ...], ...] = ()

        if self._evidence and self._embedding_weight > 0:
            assert self._embedder is not None
            self._document_embeddings = self._encode(
                [item.text for item in self._evidence]
            )

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        return tuple(token.lower() for token in _TOKEN_RE.findall(text))

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        assert self._embedder is not None
        vectors = tuple(tuple(float(value) for value in row) for row in self._embedder.encode(texts))
        if len(vectors) != len(texts):
            raise ValueError("Embedder must return exactly one vector per input text.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1 or (vectors and 0 in dimensions):
            raise ValueError("Embedding vectors must have the same non-zero dimension.")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embedding vectors must contain only finite values.")
        return vectors

    def _bm25_score(self, query_tokens: Sequence[str], document_index: int) -> float:
        if not query_tokens or not self._evidence:
            return 0.0

        term_frequencies = Counter(self._tokens[document_index])
        document_length = self._document_lengths[document_index]
        average_length = self._average_document_length or 1.0
        score = 0.0

        for token, query_frequency in Counter(query_tokens).items():
            frequency = term_frequencies[token]
            if frequency == 0:
                continue
            document_frequency = self._document_frequency[token]
            inverse_document_frequency = math.log1p(
                (len(self._evidence) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = self._k1 * (
                1 - self._b + self._b * document_length / average_length
            )
            term_score = inverse_document_frequency * (
                frequency * (self._k1 + 1) / (frequency + length_normalization)
            )
            score += query_frequency * term_score

        return score

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Query and indexed embedding dimensions do not match.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return min(1.0, max(0.0, (cosine + 1.0) / 2.0))

    def retrieve(self, claim: Claim, top_k: int = 5) -> tuple[RetrievedEvidence, ...]:
        """Return the highest-ranked evidence passages for ``claim``."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self._evidence:
            return ()

        query_tokens = self._tokenize(claim.text)
        bm25_scores = tuple(
            self._bm25_score(query_tokens, index)
            for index in range(len(self._evidence))
        )
        max_bm25 = max(bm25_scores, default=0.0)
        normalized_bm25 = tuple(
            score / max_bm25 if max_bm25 > 0 else 0.0 for score in bm25_scores
        )

        embedding_scores = [0.0] * len(self._evidence)
        if self._embedding_weight > 0:
            assert self._embedder is not None
            query_embedding = self._encode((claim.text,))[0]
            embedding_scores = [
                self._cosine_similarity(query_embedding, document_embedding)
                for document_embedding in self._document_embeddings
            ]

        combined_scores = tuple(
            self._bm25_weight * normalized_bm25[index]
            + self._embedding_weight * embedding_scores[index]
            for index in range(len(self._evidence))
        )
        ordered_indices = sorted(
            range(len(self._evidence)),
            key=lambda index: (-combined_scores[index], index),
        )[:top_k]

        return tuple(
            RetrievedEvidence(
                evidence=self._evidence[index],
                bm25_score=bm25_scores[index],
                embedding_score=embedding_scores[index],
                combined_score=combined_scores[index],
                rank=rank,
            )
            for rank, index in enumerate(ordered_indices, start=1)
        )
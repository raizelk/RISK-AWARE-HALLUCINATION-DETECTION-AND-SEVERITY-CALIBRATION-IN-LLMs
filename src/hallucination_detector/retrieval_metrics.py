"""Metrics for evaluating evidence retrieval against labeled relevant IDs."""

from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1.")


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Return the fraction of relevant evidence found in the first ``k`` hits."""
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Return the fraction of the first ``k`` ranked IDs that are relevant.

    The denominator is always ``k`` (standard Precision@K), including when
    fewer than ``k`` documents were returned.
    """
    _validate_k(k)
    relevant = set(relevant_ids)
    return sum(item_id in relevant for item_id in retrieved_ids[:k]) / k


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """Return the reciprocal rank of the first relevant result for one query."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_by_query: Sequence[Collection[str]],
) -> float:
    """Average reciprocal rank across queries with corresponding gold IDs."""
    if len(rankings) != len(relevant_by_query):
        raise ValueError("Provide one relevant-ID collection for each query ranking.")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_by_query)
    ) / len(rankings)
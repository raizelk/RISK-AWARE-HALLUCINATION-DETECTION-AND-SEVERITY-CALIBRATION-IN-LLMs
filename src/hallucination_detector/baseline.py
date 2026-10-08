"""A deterministic baseline for validating the initial project contract.

This is intentionally simple. It provides a replaceable interface for later
BM25, dense retrieval, semantic similarity, and NLI implementations.
"""

import re
from collections import Counter

from .domain import Claim, Evidence, VerificationLabel, VerificationResult

_ATOMIC_SPLIT_RE = re.compile(
    r"""
    (?<=[.!?])\s+
    | \s*[;:]\s*
    | \s+(?:and|but|however|although|though|while|whereas|because|so|yet|or|nor)\s+
    | \s*(?:--|—)\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def extract_claims(response: str) -> tuple[Claim, ...]:
    """Split a response into atomic factual claims while preserving offsets."""
    if not response.strip():
        return ()

    parts = [part.strip() for part in _ATOMIC_SPLIT_RE.split(response) if part and part.strip()]
    claims: list[Claim] = []
    search_start = 0

    for index, text in enumerate(parts, start=1):
        match_start = response.find(text, search_start)
        if match_start == -1:
            match_start = response.find(text)
        if match_start == -1:
            continue
        match_end = match_start + len(text)
        claims.append(Claim(id=f"claim-{index}", text=text, start=match_start, end=match_end))
        search_start = match_end

    return tuple(claims)


def lexical_similarity(left: str, right: str) -> float:
    """Return token-set overlap, providing a transparent baseline signal."""
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def retrieve_evidence(claim: Claim, evidence: tuple[Evidence, ...], top_k: int = 3) -> tuple[Evidence, ...]:
    """Rank evidence with a small BM25-like term-frequency overlap baseline."""
    claim_tokens = Counter(re.findall(r"[a-z0-9]+", claim.text.lower()))
    ranked = sorted(
        evidence,
        key=lambda item: sum(claim_tokens[token] for token in re.findall(r"[a-z0-9]+", item.text.lower())),
        reverse=True,
    )
    return tuple(ranked[:top_k])


def verify_claim(claim: Claim, evidence: tuple[Evidence, ...], support_threshold: float = 0.25) -> VerificationResult:
    """Classify a claim using transparent lexical evidence signals."""
    ranked = retrieve_evidence(claim, evidence)
    best_similarity = max((lexical_similarity(claim.text, item.text) for item in ranked), default=0.0)
    if best_similarity > support_threshold:
        label = VerificationLabel.SUPPORTED
        explanation = "The claim shares sufficient lexical evidence with the retrieved source."
    else:
        label = VerificationLabel.UNKNOWN
        explanation = "No retrieved source provides enough lexical overlap to verify the claim."
    return VerificationResult(claim, label, ranked, best_similarity, explanation)
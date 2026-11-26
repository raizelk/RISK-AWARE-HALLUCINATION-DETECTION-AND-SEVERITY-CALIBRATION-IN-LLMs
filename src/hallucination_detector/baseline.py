"""A deterministic baseline for validating the initial project contract.

This is intentionally simple. It provides a replaceable interface for later
BM25, dense retrieval, semantic similarity, and NLI implementations.
"""

import re
from collections import Counter

from .domain import Claim, Evidence, VerificationLabel, VerificationResult

_ATOMIC_SPLIT_RE = re.compile(
    r"""(?<=[.!?])(["'”’)\]]*)\s+|[;:]\s*|(?<=\n)\s*(?=(?:[-*+]\s+|\d+[.)]\s+))""",
)
_CLAUSE_JOIN_RE = re.compile(
    r"\s+(?:and|but|however|although|though|while|whereas|because|so|yet)\s+",
    flags=re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?")
_INTRO_PREFIX_RE = re.compile(
    r"^(?:here\s+(?:are|is)|the following\s+(?:are|is)|below\s+(?:are|is)|"
    r"in summary|to summarize)\b",
    flags=re.IGNORECASE,
)
_INCOMPLETE_END_RE = re.compile(
    r"\b(?:a|an|the|of|to|for|by|in|on|at|with|from|and|or|but|which|who|that)\s*$",
    flags=re.IGNORECASE,
)
_INCOMPLETE_MODIFIER_END_RE = re.compile(
    r"\b(?:a|an|the)\s+(?:[a-z]+ed|[a-z]+ing)\s*$",
    flags=re.IGNORECASE,
)
_MARKDOWN_HEADING_RE = re.compile(
    r"\s*(?:[-*+]\s+|\d+[.)]\s*)?\*\*[^*]+\*\*\s*[.!?]?\s*"
)
_LIST_PREFIX_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_AUXILIARY_VERBS = frozenset(
    "am is are was were be been being have has had do does did can could will would "
    "shall should may might must".split()
)
_LEXICAL_VERBS = frozenset(
    """
    achieve affect appear apply avoid become build call cause contain contribute control
    create depend develop differ die delay ensure enter experience explain form follow
    freeze give help improve include increase involve keep lead last live lower make monitor
    move need offer occur play prevent produce protect provide range receive recommend
    reduce refer remain remember result recognize require spread strengthen stimulate
    support take train treat trigger use vary work
    """.split()
)


def extract_claims(response: str) -> tuple[Claim, ...]:
    """Extract complete statement-like claims while preserving response offsets."""
    if not response.strip():
        return ()

    parts: list[tuple[str, int]] = []
    start = 0
    for separator in _ATOMIC_SPLIT_RE.finditer(response):
        parts.append(
            (response[start:separator.start()] + (separator.group(1) or ""), start)
        )
        start = separator.end()
    parts.append((response[start:], start))

    claims: list[Claim] = []
    for part, part_start in parts:
        for text, text_start in _split_complete_clauses(part, part_start):
            trimmed = text.strip()
            if not trimmed:
                continue
            leading_space = len(text) - len(text.lstrip())
            match_start = text_start + leading_space
            list_prefix = _LIST_PREFIX_RE.match(trimmed)
            if list_prefix is not None:
                remainder = trimmed[list_prefix.end() :]
                following_space = len(remainder) - len(remainder.lstrip())
                match_start += list_prefix.end() + following_space
                trimmed = remainder.lstrip()
            match_end = match_start + len(trimmed)
            if _is_complete_statement(trimmed):
                claims.append(
                    Claim(
                        id=f"claim-{len(claims) + 1}",
                        text=trimmed,
                        start=match_start,
                        end=match_end,
                    )
                )

    return tuple(claims)


def _split_complete_clauses(text: str, offset: int) -> list[tuple[str, int]]:
    for separator in _CLAUSE_JOIN_RE.finditer(text):
        left = text[:separator.start()]
        right_start = separator.end()
        right = text[right_start:]
        if (
            _is_complete_statement(left.strip())
            and _is_complete_statement(right.strip())
            and right.lstrip()[:1].isupper()
        ):
            return _split_complete_clauses(left, offset) + _split_complete_clauses(
                right, offset + right_start
            )
    return [(text, offset)]


def _is_complete_statement(text: str) -> bool:
    if _MARKDOWN_HEADING_RE.fullmatch(text):
        return False
    normalized = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", text.strip())
    normalized = re.sub(r"[*_`]", "", normalized)
    if _INTRO_PREFIX_RE.match(normalized):
        return False

    words = [match.group().lower() for match in _WORD_RE.finditer(normalized)]
    content = normalized.rstrip(" .!?")
    if len(words) < 3 or _INCOMPLETE_END_RE.search(content):
        return False
    if _INCOMPLETE_MODIFIER_END_RE.search(content):
        return False

    return any(
        word in _AUXILIARY_VERBS
        or word in _LEXICAL_VERBS
        or word.endswith("s") and word[:-1] in _LEXICAL_VERBS
        or word.endswith("es") and word[:-2] in _LEXICAL_VERBS
        or word.endswith("ies") and f"{word[:-3]}y" in _LEXICAL_VERBS
        or word.endswith("ed") and word[:-2] in _LEXICAL_VERBS
        for word in words[1:-1]
    )


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
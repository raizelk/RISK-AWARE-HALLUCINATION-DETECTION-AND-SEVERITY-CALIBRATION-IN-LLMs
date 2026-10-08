"""Claim-level explanations and safe response highlighting."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Optional, Sequence

from .domain import Claim, Evidence, Severity, VerificationLabel, VerificationResult
from .severity import SeverityAssessment


@dataclass(frozen=True)
class ClaimExplanation:
    """Verification details associated with one span in a response."""

    claim: Claim
    verification_label: VerificationLabel
    severity: Optional[Severity]
    evidence: tuple[Evidence, ...]
    verification_reason: str
    severity_reason: Optional[str]


@dataclass(frozen=True)
class ResponseExplanation:
    """A response annotated with claim highlights and per-claim explanations."""

    original_response: str
    highlighted_response_html: str
    claims: tuple[ClaimExplanation, ...]


def explain_response(
    original_response: str,
    verifications: Sequence[VerificationResult],
    severity_assessments: Sequence[SeverityAssessment] = (),
) -> ResponseExplanation:
    """Highlight verified claim spans and retain their evidence and reasons.

    Claim offsets must identify the exact claim text in ``original_response``.
    Overlapping claim spans are rejected rather than rendered ambiguously.
    """
    ordered_verifications = tuple(
        sorted(verifications, key=lambda result: (result.claim.start, result.claim.end))
    )
    by_claim_id: dict[str, VerificationResult] = {}
    previous_end = 0

    for result in ordered_verifications:
        claim = result.claim
        _validate_claim_span(original_response, claim)
        if claim.id in by_claim_id:
            raise ValueError(f"Duplicate claim ID in verification results: {claim.id!r}.")
        if claim.start < previous_end:
            raise ValueError("Claim spans must not overlap.")
        by_claim_id[claim.id] = result
        previous_end = claim.end

    severity_by_claim_id: dict[str, SeverityAssessment] = {}
    for assessment in severity_assessments:
        claim_id = assessment.verification.claim.id
        if claim_id in severity_by_claim_id:
            raise ValueError(f"Duplicate severity assessment for claim {claim_id!r}.")
        result = by_claim_id.get(claim_id)
        if result is None or result != assessment.verification:
            raise ValueError(
                f"Severity assessment for claim {claim_id!r} does not match a "
                "verification result."
            )
        severity_by_claim_id[claim_id] = assessment

    chunks: list[str] = []
    explanations: list[ClaimExplanation] = []
    cursor = 0

    for result in ordered_verifications:
        claim = result.claim
        chunks.append(escape(original_response[cursor:claim.start]))
        severity_assessment = severity_by_claim_id.get(claim.id)
        attributes = (
            f' data-claim-id="{escape(claim.id, quote=True)}"'
            f' data-verification="{escape(result.label.value, quote=True)}"'
        )
        if severity_assessment is not None and severity_assessment.severity is not None:
            attributes += (
                f' data-severity="{escape(severity_assessment.severity.value, quote=True)}"'
            )
        chunks.append(
            f"<mark{attributes}>{escape(original_response[claim.start:claim.end])}</mark>"
        )
        cursor = claim.end
        explanations.append(
            ClaimExplanation(
                claim=claim,
                verification_label=result.label,
                severity=severity_assessment.severity if severity_assessment else None,
                evidence=result.evidence,
                verification_reason=result.explanation,
                severity_reason=severity_assessment.rationale if severity_assessment else None,
            )
        )

    chunks.append(escape(original_response[cursor:]))
    return ResponseExplanation(
        original_response=original_response,
        highlighted_response_html="".join(chunks),
        claims=tuple(explanations),
    )


def _validate_claim_span(response: str, claim: Claim) -> None:
    if claim.start < 0 or claim.end > len(response) or claim.start >= claim.end:
        raise ValueError(
            f"Claim {claim.id!r} has an invalid response span "
            f"[{claim.start}, {claim.end}) for response length {len(response)}."
        )
    if response[claim.start:claim.end] != claim.text:
        raise ValueError(
            f"Claim {claim.id!r} text does not match its response span "
            f"[{claim.start}, {claim.end})."
        )

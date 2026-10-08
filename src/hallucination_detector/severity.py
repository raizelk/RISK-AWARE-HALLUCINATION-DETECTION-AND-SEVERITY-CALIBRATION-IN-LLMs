"""Rule-based severity assessment for unsupported and contradicted claims."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .domain import Severity, VerificationLabel, VerificationResult

_RISK_DOMAIN_PATTERNS = (
    (
        "medical",
        re.compile(
            r"\b(?:health|medical|medicine|medication|drug|dose|dosage|"
            r"prescription|diagnos\w*|treatment|surgery|vaccine|insulin|"
            r"symptom|doctor|patient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "safety",
        re.compile(
            r"\b(?:safety|unsafe|dangerous|weapon|poison|toxic|chemical\w*|"
            r"explosive|flammable|fire|electrocute|enclosed space|"
            r"self[- ]harm|suicid\w*|injur\w*|death|fatal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legal",
        re.compile(
            r"\b(?:legal|law|court|lawsuit|contract|criminal|arrest|"
            r"sue|attorney|lawyer|rights?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "finance",
        re.compile(
            r"\b(?:financ\w*|invest\w*|stock|shares|loan|debt|mortgage|"
            r"interest rate|bankruptcy|tax(?:es|ation)?|retirement fund)\b",
            re.IGNORECASE,
        ),
    ),
)

_ACTIONABLE_ADVICE_RE = re.compile(
    r"\b(?:should|must|need to|have to|recommend(?:s|ed|ing)?|"
    r"advise(?:s|d|ing)?|instruct(?:s|ed|ing)?|"
    r"take|stop|start|increase|decrease|double|halve|mix|consume|"
    r"sign|transfer|invest|pay|disclose|avoid|use)\b"
    r"|\bdo not\b|\bdon't\b",
    re.IGNORECASE,
)

_SEVERITY_ORDER = (
    Severity.MILD,
    Severity.MODERATE,
    Severity.SEVERE,
    Severity.CRITICAL,
)
_HIGH_CONTRADICTION_THRESHOLD = 0.8


@dataclass(frozen=True)
class SeverityAssessment:
    """Severity decision and the signals that explain it."""

    verification: VerificationResult
    severity: Optional[Severity]
    risk_domains: tuple[str, ...]
    rationale: str


class SeverityClassifier:
    """Apply a transparent impact policy to one claim verification result.

    The numeric cutoffs are initial policy defaults, not learned or calibrated
    probabilities. They should be reviewed against a labeled validation set.
    """

    def __init__(self, high_contradiction_threshold: float = _HIGH_CONTRADICTION_THRESHOLD) -> None:
        if (
            not math.isfinite(high_contradiction_threshold)
            or not 0.0 <= high_contradiction_threshold <= 1.0
        ):
            raise ValueError("high_contradiction_threshold must be finite and in [0, 1].")
        self._high_contradiction_threshold = high_contradiction_threshold

    def classify(
        self,
        verification: VerificationResult,
        *,
        centrality: float = 0.0,
    ) -> SeverityAssessment:
        """Assess severity; supported claims receive no error severity.

        ``centrality`` is an optional caller-provided estimate in [0, 1] of
        how important the claim is to the response's main answer.
        """
        if not math.isfinite(centrality) or not 0.0 <= centrality <= 1.0:
            raise ValueError("centrality must be finite and in [0, 1].")
        contradiction_score = verification.contradiction_score
        if not math.isfinite(contradiction_score) or not 0.0 <= contradiction_score <= 1.0:
            raise ValueError("verification contradiction_score must be finite and in [0, 1].")

        if verification.label is VerificationLabel.SUPPORTED:
            return SeverityAssessment(
                verification=verification,
                severity=None,
                risk_domains=(),
                rationale="The claim is supported; no error severity is assigned.",
            )

        text = verification.claim.text
        risk_domains = tuple(
            name for name, pattern in _RISK_DOMAIN_PATTERNS if pattern.search(text)
        )
        actionable_advice = bool(_ACTIONABLE_ADVICE_RE.search(text))
        rationale_parts = []

        if verification.label is VerificationLabel.CONTRADICTED:
            severity = Severity.MODERATE
            rationale_parts.append("Contradicted claims start at moderate severity.")
        elif verification.label is VerificationLabel.UNKNOWN:
            severity = Severity.MILD
            rationale_parts.append("Unknown claims start at mild severity.")
        else:
            raise ValueError(f"Unsupported verification label: {verification.label!r}")

        if (
            verification.label is VerificationLabel.CONTRADICTED
            and contradiction_score >= self._high_contradiction_threshold
        ):
            severity = self._escalate(severity)
            rationale_parts.append(
                f"Strong contradiction score ({contradiction_score:.2f}) raises severity."
            )

        if centrality >= 0.75:
            escalated = self._escalate(severity)
            if escalated is severity:
                rationale_parts.append(
                    f"High response centrality ({centrality:.2f}) was considered; "
                    "severity escalation is capped at severe unless the claim contains "
                    "actionable high-impact advice."
                )
            else:
                severity = escalated
                rationale_parts.append(
                    f"High response centrality ({centrality:.2f}) raises severity."
                )

        if risk_domains and centrality >= 0.5 and severity in {
            Severity.MILD,
            Severity.MODERATE,
        }:
            severity = Severity.SEVERE
            rationale_parts.append(
                "A high-impact domain is central to the response, raising severity "
                "to severe."
            )

        if risk_domains and actionable_advice:
            severity = Severity.CRITICAL
            rationale_parts.append(
                "Unsupported actionable advice in a high-impact domain is critical."
            )

        if risk_domains:
            rationale_parts.append(f"Detected high-impact domain(s): {', '.join(risk_domains)}.")
        return SeverityAssessment(
            verification=verification,
            severity=severity,
            risk_domains=risk_domains,
            rationale=" ".join(rationale_parts),
        )

    @staticmethod
    def _escalate(severity: Severity) -> Severity:
        index = _SEVERITY_ORDER.index(severity)
        return _SEVERITY_ORDER[min(index + 1, len(_SEVERITY_ORDER) - 2)]

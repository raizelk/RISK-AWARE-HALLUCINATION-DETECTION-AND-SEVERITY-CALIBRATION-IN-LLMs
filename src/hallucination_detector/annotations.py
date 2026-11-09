"""Validation and evaluation mappings for human-reviewed claim annotations."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import extract_claims
from .domain import ModelResponse, Severity, VerificationLabel

ClaimKey = tuple[str, str, str]


@dataclass(frozen=True)
class ClaimAnnotation:
    annotator_id: str
    annotation_status: str
    model_name: str
    dataset_item_id: str
    claim_id: str
    claim_text: str
    claim_start: int
    claim_end: int
    verification_label: VerificationLabel
    evidence_ids: tuple[str, ...]
    severity: Severity | None
    rationale: str

    @property
    def key(self) -> ClaimKey:
        return (self.model_name, self.dataset_item_id, self.claim_id)


@dataclass(frozen=True)
class GoldAnnotations:
    labels: Mapping[ClaimKey, VerificationLabel]
    evidence_ids: Mapping[ClaimKey, Collection[str]]
    severity: Mapping[ClaimKey, Severity | None]
    annotations: tuple[ClaimAnnotation, ...]


@dataclass(frozen=True)
class AnnotationAgreement:
    compared_claims: int
    exact_agreement: float
    cohen_kappa: float | None
    disagreement_keys: tuple[ClaimKey, ...]


def load_adjudicated_annotations(
    path: str | Path,
    responses: Sequence[ModelResponse],
    known_evidence_ids: Collection[str],
    *,
    require_complete: bool = False,
) -> GoldAnnotations:
    """Load final labels after validating claim identity, spans, and evidence IDs."""
    annotations = _load_validated(
        path,
        responses,
        known_evidence_ids,
        required_status="adjudicated",
    )
    by_key = {annotation.key: annotation for annotation in annotations}
    if require_complete:
        expected = set(_expected_claims(responses))
        if by_key.keys() != expected:
            missing = sorted(expected - by_key.keys())
            extra = sorted(by_key.keys() - expected)
            raise ValueError(
                f"Adjudicated annotations are incomplete or unexpected; missing={missing!r}, extra={extra!r}."
            )

    return GoldAnnotations(
        labels={key: item.verification_label for key, item in by_key.items()},
        evidence_ids={
            key: frozenset(item.evidence_ids)
            for key, item in by_key.items()
            if item.evidence_ids
        },
        severity={key: item.severity for key, item in by_key.items()},
        annotations=annotations,
    )


def compare_independent_annotations(
    first_path: str | Path,
    second_path: str | Path,
    responses: Sequence[ModelResponse],
    known_evidence_ids: Collection[str],
) -> AnnotationAgreement:
    """Compare two complete independent passes using label agreement and Cohen's kappa."""
    first = _load_validated(
        first_path, responses, known_evidence_ids, required_status="independent"
    )
    second = _load_validated(
        second_path, responses, known_evidence_ids, required_status="independent"
    )
    if not first or not second:
        raise ValueError("Both independent annotation files must contain claim labels.")
    if first[0].annotator_id == second[0].annotator_id:
        raise ValueError("Independent annotation files must have different annotator IDs.")

    first_by_key = {item.key: item for item in first}
    second_by_key = {item.key: item for item in second}
    if first_by_key.keys() != second_by_key.keys():
        raise ValueError("Independent annotation files must cover the same claim keys.")

    keys = tuple(first_by_key)
    agreements = sum(
        first_by_key[key].verification_label == second_by_key[key].verification_label
        for key in keys
    )
    exact_agreement = agreements / len(keys)
    first_counts = Counter(item.verification_label for item in first_by_key.values())
    second_counts = Counter(item.verification_label for item in second_by_key.values())
    expected_agreement = sum(
        first_counts[label] * second_counts[label] for label in VerificationLabel
    ) / (len(keys) ** 2)
    kappa = (
        None
        if expected_agreement == 1.0
        else (exact_agreement - expected_agreement) / (1.0 - expected_agreement)
    )
    disagreements = tuple(
        key
        for key in keys
        if first_by_key[key].verification_label != second_by_key[key].verification_label
    )
    return AnnotationAgreement(len(keys), exact_agreement, kappa, disagreements)


def _load_validated(
    path: str | Path,
    responses: Sequence[ModelResponse],
    known_evidence_ids: Collection[str],
    *,
    required_status: str,
) -> tuple[ClaimAnnotation, ...]:
    source = Path(path)
    expected_claims = _expected_claims(responses)
    records: list[ClaimAnnotation] = []
    seen: set[ClaimKey] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid annotation JSON on line {line_number}.") from error
            if not isinstance(value, dict):
                raise ValueError(f"Annotation line {line_number} must be a JSON object.")
            annotation = _parse_record(value, line_number, known_evidence_ids)
            if annotation.annotation_status != required_status:
                raise ValueError(
                    f"Annotation line {line_number} must have status {required_status!r}."
                )
            expected_claim = expected_claims.get(annotation.key)
            if expected_claim is None:
                raise ValueError(f"Annotation line {line_number} references an unknown claim.")
            if (
                annotation.claim_text != expected_claim.text
                or annotation.claim_start != expected_claim.start
                or annotation.claim_end != expected_claim.end
            ):
                raise ValueError(
                    f"Annotation line {line_number} does not match the extracted claim text and offsets."
                )
            if annotation.key in seen:
                raise ValueError(f"Duplicate annotation for claim key {annotation.key!r}.")
            seen.add(annotation.key)
            records.append(annotation)
    annotator_ids = {annotation.annotator_id for annotation in records}
    if len(annotator_ids) > 1:
        raise ValueError(f"{source.name} must contain annotations from exactly one annotator.")
    return tuple(records)


def _parse_record(
    value: Mapping[str, Any],
    line_number: int,
    known_evidence_ids: Collection[str],
) -> ClaimAnnotation:
    if value.get("schema_version") != 1:
        raise ValueError(f"Annotation line {line_number} has an unsupported schema_version.")
    annotator_id = _required_text(value, "annotator_id", line_number)
    status = _required_text(value, "annotation_status", line_number)
    if value.get("claim_review_status") != "accepted":
        raise ValueError(
            f"Annotation line {line_number}: claim must be reviewed and marked 'accepted' first."
        )
    model_name = _required_text(value, "model_name", line_number)
    item_id = _required_text(value, "dataset_item_id", line_number)
    claim_id = _required_text(value, "claim_id", line_number)
    claim_text = _required_text(value, "claim_text", line_number)
    claim_start = _required_offset(value, "claim_start", line_number)
    claim_end = _required_offset(value, "claim_end", line_number)
    rationale = _required_text(value, "rationale", line_number)

    raw_label = value.get("verification_label")
    if not isinstance(raw_label, str):
        raise ValueError(f"Annotation line {line_number} has an invalid verification_label.")
    try:
        label = VerificationLabel(raw_label)
    except ValueError as error:
        raise ValueError(f"Annotation line {line_number} has an invalid verification_label.") from error

    raw_severity = value.get("severity")
    if raw_severity is None:
        severity = None
    else:
        if not isinstance(raw_severity, str):
            raise ValueError(f"Annotation line {line_number} has an invalid severity.")
        try:
            severity = Severity(raw_severity)
        except ValueError as error:
            raise ValueError(f"Annotation line {line_number} has an invalid severity.") from error
    if label is VerificationLabel.SUPPORTED and severity is not None:
        raise ValueError(f"Annotation line {line_number}: supported claims must have null severity.")
    if label is not VerificationLabel.SUPPORTED and severity is None:
        raise ValueError(f"Annotation line {line_number}: unsupported claims require a severity label.")

    raw_evidence_ids = value.get("evidence_ids")
    if not isinstance(raw_evidence_ids, list) or any(
        not isinstance(evidence_id, str) or not evidence_id.strip()
        for evidence_id in raw_evidence_ids
    ):
        raise ValueError(f"Annotation line {line_number}: evidence_ids must be an array of IDs.")
    if len(set(raw_evidence_ids)) != len(raw_evidence_ids):
        raise ValueError(f"Annotation line {line_number}: evidence_ids contains duplicates.")
    unknown_ids = set(raw_evidence_ids) - set(known_evidence_ids)
    if unknown_ids:
        raise ValueError(f"Annotation line {line_number}: unknown evidence IDs {sorted(unknown_ids)!r}.")

    return ClaimAnnotation(
        annotator_id=annotator_id,
        annotation_status=status,
        model_name=model_name,
        dataset_item_id=item_id,
        claim_id=claim_id,
        claim_text=claim_text,
        claim_start=claim_start,
        claim_end=claim_end,
        verification_label=label,
        evidence_ids=tuple(raw_evidence_ids),
        severity=severity,
        rationale=rationale,
    )


def _expected_claims(responses: Sequence[ModelResponse]) -> dict[ClaimKey, Any]:
    expected: dict[ClaimKey, Any] = {}
    response_keys: set[tuple[str, str]] = set()
    for response in responses:
        response_key = (response.model_name, response.dataset_item_id)
        if response_key in response_keys:
            raise ValueError(f"Duplicate response for model/item pair {response_key!r}.")
        response_keys.add(response_key)
        for claim in extract_claims(response.text):
            expected[(response.model_name, response.dataset_item_id, claim.id)] = claim
    return expected


def _required_text(record: Mapping[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Annotation line {line_number}: {field} must be non-empty text.")
    return value


def _required_offset(record: Mapping[str, Any], field: str, line_number: int) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"Annotation line {line_number}: {field} must be a non-negative integer.")
    return value
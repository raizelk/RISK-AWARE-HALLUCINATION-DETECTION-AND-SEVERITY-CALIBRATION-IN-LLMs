"""Tune detector thresholds and report dev-set metrics from saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.annotations import load_adjudicated_annotations
from hallucination_detector.calibration import (
    CalibrationObservation,
    SeverityCalibrationObservation,
    calibrate_severity_threshold,
    calibrate_verification_thresholds,
    evaluate_severity,
)
from hallucination_detector.domain import Claim, Severity, VerificationLabel, VerificationResult
from hallucination_detector.generation import JsonlResponseStore
from hallucination_detector.retrieval_metrics import recall_at_k
from hallucination_detector.runner import load_benchmark, write_json_atomic
from hallucination_detector.severity import SeverityClassifier
from hallucination_detector.verification import classify_verification_scores


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _read_evidence_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid evidence JSON on line {line_number}.") from error
            evidence_id = record.get("evidence_id") if isinstance(record, dict) else None
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError(f"Evidence line {line_number} has no valid evidence_id.")
            if evidence_id in ids:
                raise ValueError(f"Duplicate evidence ID {evidence_id!r}.")
            ids.add(evidence_id)
    return ids


def _report_claims(report_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    report = _read_json(report_path)
    if report.get("schema_version") != 1 or not isinstance(report.get("responses"), list):
        raise ValueError("Evaluation report must use schema_version 1 and contain responses.")
    claims: dict[tuple[str, str, str], dict[str, Any]] = {}
    for evaluated_response in report["responses"]:
        response = evaluated_response.get("response", {})
        model_name = response.get("model_name")
        item_id = response.get("dataset_item_id")
        for evaluated_claim in evaluated_response.get("claims", []):
            verification = evaluated_claim.get("verification", {})
            claim = verification.get("claim", {})
            key = (model_name, item_id, claim.get("id"))
            if not all(isinstance(part, str) and part for part in key):
                raise ValueError("Evaluation report contains an invalid model/item/claim key.")
            if key in claims:
                raise ValueError(f"Evaluation report contains duplicate claim key {key!r}.")
            retrieved = evaluated_claim.get("retrieved_evidence", [])
            claims[key] = {
                "claim": Claim(
                    id=claim["id"],
                    text=claim["text"],
                    start=claim["start"],
                    end=claim["end"],
                ),
                "support_score": verification["support_score"],
                "contradiction_score": verification["contradiction_score"],
                "retrieved_ids": [
                    item.get("evidence", {}).get("id", "") for item in retrieved
                ],
            }
    return claims


def _threshold_grid(step: float) -> tuple[float, ...]:
    if not math.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("threshold_step must be finite and in (0, 1].")
    values = [round(index * step, 10) for index in range(math.floor(1.0 / step) + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return tuple(values)


def calibrate(
    report_path: Path,
    responses_path: Path,
    annotations_path: Path,
    prompts_path: Path,
    evidence_path: Path,
    split_path: Path,
    output_path: Path,
    *,
    threshold_step: float = 0.05,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite calibration artifact {output_path}.")
    benchmark = load_benchmark(prompts_path, evidence_path)
    if not benchmark.is_frozen:
        raise ValueError("Freeze the reviewed benchmark before fitting calibration thresholds.")
    split = _read_json(split_path)
    if split.get("schema_version") != 1:
        raise ValueError("Topic split manifest must use schema_version 1.")
    if split.get("prompt_sha256") != benchmark.prompt_sha256:
        raise ValueError("Topic split prompt hash does not match the frozen benchmark.")
    if split.get("evidence_sha256") != benchmark.evidence_sha256:
        raise ValueError("Topic split evidence hash does not match the frozen benchmark.")
    item_splits = split.get("item_splits")
    if not isinstance(item_splits, dict):
        raise ValueError("Topic split manifest is missing item_splits.")
    expected_item_ids = {prompt.dataset_item_id for prompt in benchmark.prompts}
    if set(item_splits) != expected_item_ids:
        raise ValueError("Topic split does not cover exactly the frozen benchmark items.")
    if any(value not in {"train", "dev", "test"} for value in item_splits.values()):
        raise ValueError("Topic split contains an invalid split name.")
    topic_splits = split.get("topic_splits")
    if not isinstance(topic_splits, dict) or any(
        value not in {"train", "dev", "test"} for value in topic_splits.values()
    ):
        raise ValueError("Topic split manifest contains invalid topic assignments.")

    responses = JsonlResponseStore().load(responses_path)
    if not responses:
        raise ValueError("Response JSONL contains no model responses.")
    evidence_ids = _read_evidence_ids(evidence_path)
    gold = load_adjudicated_annotations(
        annotations_path,
        responses,
        evidence_ids,
        require_complete=True,
    )
    report_claims = _report_claims(report_path)
    expected_keys = set(gold.labels)
    if set(report_claims) != expected_keys:
        raise ValueError("Evaluation report claim keys do not match the adjudicated responses.")

    dev_item_ids = {item_id for item_id, name in item_splits.items() if name == "dev"}
    if not dev_item_ids:
        raise ValueError("Topic split contains no development items.")
    dev_keys = {key for key in expected_keys if key[1] in dev_item_ids}
    if not dev_keys:
        raise ValueError("No adjudicated claims are available for the development items.")

    observations = tuple(
        CalibrationObservation(
            key=key,
            support_score=report_claims[key]["support_score"],
            contradiction_score=report_claims[key]["contradiction_score"],
            gold_label=gold.labels[key],
        )
        for key in sorted(dev_keys)
    )
    thresholds = _threshold_grid(threshold_step)
    verification_calibration = calibrate_verification_thresholds(
        observations,
        candidate_thresholds=thresholds,
        objective="macro_f1",
    )

    severity_observations = tuple(
        SeverityCalibrationObservation(
            key=key,
            claim=report_claims[key]["claim"],
            verification_label=classify_verification_scores(
                report_claims[key]["support_score"],
                report_claims[key]["contradiction_score"],
                support_threshold=verification_calibration.support_threshold,
                contradiction_threshold=verification_calibration.contradiction_threshold,
            ),
            contradiction_score=report_claims[key]["contradiction_score"],
            gold_severity=gold.severity[key],
        )
        for key in sorted(dev_keys)
    )
    severity_calibration = calibrate_severity_threshold(
        severity_observations,
        candidate_thresholds=thresholds,
    )
    severity_predictions = {
        item.key: SeverityClassifier(high_contradiction_threshold=severity_calibration.high_contradiction_threshold)
        .classify(
            VerificationResult(
                claim=item.claim,
                label=item.verification_label,
                contradiction_score=item.contradiction_score,
            )
        )
        .severity
        for item in severity_observations
    }
    severity_gold = {key: gold.severity[key] for key in dev_keys}
    severity_metrics = evaluate_severity(severity_gold, severity_predictions)

    retrieval_recalls: dict[str, float | None] = {}
    for k in (1, 3, 5):
        scores = [
            recall_at_k(report_claims[key]["retrieved_ids"], gold.evidence_ids[key], k)
            for key in dev_keys & gold.evidence_ids.keys()
        ]
        retrieval_recalls[str(k)] = sum(scores) / len(scores) if scores else None

    dev_topics = sorted(topic for topic, name in topic_splits.items() if name == "dev")
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_item_ids": sorted(dev_item_ids),
        "development_topics": dev_topics,
        "development_claim_count": len(dev_keys),
        "verification": {
            "support_threshold": verification_calibration.support_threshold,
            "contradiction_threshold": verification_calibration.contradiction_threshold,
            "objective": verification_calibration.objective,
            "objective_score": verification_calibration.objective_score,
            "accuracy": verification_calibration.accuracy,
            "macro_f1": verification_calibration.macro_f1,
            "balanced_accuracy": verification_calibration.balanced_accuracy,
            "coverage": verification_calibration.coverage,
            "sample_size": verification_calibration.sample_size,
            "candidates_evaluated": verification_calibration.candidates_evaluated,
        },
        "severity": {
            "high_contradiction_threshold": severity_calibration.high_contradiction_threshold,
            "accuracy": severity_metrics.accuracy,
            "macro_f1": severity_metrics.macro_f1,
            "quadratic_weighted_kappa": severity_metrics.quadratic_weighted_kappa,
            "sample_size": severity_metrics.sample_size,
            "labels": [severity.value for severity in Severity],
            "confusion_matrix": severity_metrics.confusion_matrix,
        },
        "retrieval_recall_at_k": retrieval_recalls,
        "provenance": {
            "prompt_sha256": benchmark.prompt_sha256,
            "evidence_sha256": benchmark.evidence_sha256,
            "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "evaluation_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "annotations_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        },
    }
    write_json_atomic(output_path, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "data/pilot-evaluation-report.json")
    parser.add_argument("--responses", type=Path, default=ROOT / "data/pilot-responses.jsonl")
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/annotations/adjudicated.jsonl"
    )
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/healthcare/prompts.jsonl")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data/healthcare/topic-split.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/calibration-dev.json")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        output = calibrate(
            args.report,
            args.responses,
            args.annotations,
            args.prompts,
            args.evidence,
            args.split,
            args.output,
            threshold_step=args.threshold_step,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote dev-only calibration artifact: {args.output}")
    print(
        "Verification thresholds: "
        f"support={output['verification']['support_threshold']:.2f}, "
        f"contradiction={output['verification']['contradiction_threshold']:.2f}"
    )
    print(f"Dev macro-F1: {output['verification']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Preflight and execute the held-out test-split evaluation with locked thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.annotations import load_adjudicated_annotations
from hallucination_detector.generation import JsonlResponseStore
from scripts.run_real_evaluation import main as run_evaluation


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_locked_test(
    prompts_path: Path,
    evidence_path: Path,
    benchmark_manifest_path: Path,
    split_path: Path,
    calibration_path: Path,
    responses_path: Path,
    annotations_path: Path,
) -> dict[str, float]:
    """Return locked thresholds or raise with a precise preflight error."""
    for path in (
        prompts_path,
        evidence_path,
        benchmark_manifest_path,
        split_path,
        calibration_path,
        responses_path,
        annotations_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required locked-test artifact: {path}")

    benchmark_manifest = _read_json(benchmark_manifest_path)
    if benchmark_manifest.get("schema_version") != 1:
        raise ValueError("Benchmark manifest must use schema_version 1.")
    files = benchmark_manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Benchmark manifest is missing file hashes.")
    if files.get(prompts_path.name) != _sha256(prompts_path):
        raise ValueError("Prompts differ from the frozen benchmark manifest.")
    if files.get(evidence_path.name) != _sha256(evidence_path):
        raise ValueError("Evidence differs from the frozen benchmark manifest.")

    split = _read_json(split_path)
    if split.get("schema_version") != 1:
        raise ValueError("Topic split manifest must use schema_version 1.")
    if split.get("prompt_sha256") != _sha256(prompts_path):
        raise ValueError("Topic split prompt hash does not match the frozen benchmark.")
    if split.get("evidence_sha256") != _sha256(evidence_path):
        raise ValueError("Topic split evidence hash does not match the frozen benchmark.")
    item_splits = split.get("item_splits")
    if not isinstance(item_splits, dict):
        raise ValueError("Topic split manifest is missing item_splits.")
    test_item_ids = {item_id for item_id, name in item_splits.items() if name == "test"}
    if not test_item_ids:
        raise ValueError("Topic split contains no test items.")
    if any(name not in {"train", "dev", "test"} for name in item_splits.values()):
        raise ValueError("Topic split contains an invalid split name.")

    calibration = _read_json(calibration_path)
    if calibration.get("schema_version") != 1:
        raise ValueError("Calibration artifact must use schema_version 1.")
    provenance = calibration.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Calibration artifact is missing provenance.")
    if provenance.get("prompt_sha256") != _sha256(prompts_path):
        raise ValueError("Calibration prompt hash differs from the frozen benchmark.")
    if provenance.get("evidence_sha256") != _sha256(evidence_path):
        raise ValueError("Calibration evidence hash differs from the frozen benchmark.")
    if provenance.get("split_manifest_sha256") != _sha256(split_path):
        raise ValueError("Calibration was fitted with a different topic split.")

    development_item_ids = set(calibration.get("development_item_ids", []))
    overlap = development_item_ids & test_item_ids
    if overlap:
        raise ValueError(f"Development and test item IDs overlap: {sorted(overlap)!r}.")

    verification = calibration.get("verification")
    severity = calibration.get("severity")
    if not isinstance(verification, dict) or not isinstance(severity, dict):
        raise ValueError("Calibration artifact is missing verification or severity thresholds.")

    thresholds = {
        "support_threshold": verification.get("support_threshold"),
        "contradiction_threshold": verification.get("contradiction_threshold"),
        "severity_contradiction_threshold": severity.get("high_contradiction_threshold"),
    }
    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Calibration is missing numeric {name}.")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Calibration threshold {name} is outside [0, 1].")

    responses = JsonlResponseStore().load(responses_path)
    response_ids = {response.dataset_item_id for response in responses}
    if not test_item_ids <= response_ids:
        missing = sorted(test_item_ids - response_ids)
        raise ValueError(f"Response matrix is missing test items: {missing!r}.")

    gold = load_adjudicated_annotations(
        annotations_path,
        responses,
        _read_evidence_ids(evidence_path),
        require_complete=True,
    )
    gold_test_keys = {key for key in gold.labels if key[1] in test_item_ids}
    if not gold_test_keys:
        raise ValueError("Adjudicated annotations contain no test claims.")

    return {name: float(value) for name, value in thresholds.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/healthcare/prompts.jsonl")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl")
    parser.add_argument("--benchmark-manifest", type=Path, default=ROOT / "data/healthcare/benchmark_manifest.json")
    parser.add_argument("--split", type=Path, default=ROOT / "data/healthcare/topic-split.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "data/calibration-dev.json")
    parser.add_argument("--responses", type=Path, default=ROOT / "data/responses.jsonl")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations/adjudicated.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "data/test-evaluation-report.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args(argv)

    try:
        thresholds = validate_locked_test(
            args.prompts,
            args.evidence,
            args.benchmark_manifest,
            args.split,
            args.calibration,
            args.responses,
            args.annotations,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"LOCKED TEST BLOCKED: {error}", file=sys.stderr)
        return 1

    print("LOCKED TEST PREFLIGHT PASSED")
    for name, value in thresholds.items():
        print(f"{name}={value:.4f}")

    return run_evaluation(
        [
            "--evaluate-only",
            "--prompts", str(args.prompts),
            "--evidence", str(args.evidence),
            "--responses", str(args.responses),
            "--annotations", str(args.annotations),
            "--split-manifest", str(args.split),
            "--dataset-split", "test",
            "--support-threshold", str(thresholds["support_threshold"]),
            "--contradiction-threshold", str(thresholds["contradiction_threshold"]),
            "--severity-contradiction-threshold", str(thresholds["severity_contradiction_threshold"]),
            "--top-k", str(args.top_k),
            "--nli-model", str(args.nli_model),
            "--embedding-model", str(args.embedding_model),
            "--report", str(args.report),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

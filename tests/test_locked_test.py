import hashlib
import json

import pytest

from hallucination_detector.domain import ModelResponse
from hallucination_detector.generation import JsonlResponseStore
from scripts.run_locked_test import validate_locked_test


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _paths(tmp_path):
    return (
        tmp_path / "prompts.jsonl",
        tmp_path / "evidence.jsonl",
        tmp_path / "benchmark_manifest.json",
        tmp_path / "topic-split.json",
        tmp_path / "calibration-dev.json",
        tmp_path / "responses.jsonl",
        tmp_path / "adjudicated.jsonl",
    )


def _valid_fixture(tmp_path):
    prompts, evidence, benchmark, split, calibration, responses, annotations = _paths(tmp_path)
    _write_jsonl(prompts, [{"dataset_item_id": "item-test", "topic": "topic-test"}])
    _write_jsonl(evidence, [{"evidence_id": "evidence-test", "topic": "topic-test"}])
    prompt_hash = hashlib.sha256(prompts.read_bytes()).hexdigest()
    evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()

    benchmark.write_text(json.dumps({
        "schema_version": 1,
        "files": {
            prompts.name: prompt_hash,
            evidence.name: evidence_hash,
        },
    }), encoding="utf-8")
    split.write_text(json.dumps({
        "schema_version": 1,
        "prompt_sha256": prompt_hash,
        "evidence_sha256": evidence_hash,
        "item_splits": {"item-test": "test", "item-dev": "dev", "item-train": "train"},
    }), encoding="utf-8")
    calibration.write_text(json.dumps({
        "schema_version": 1,
        "development_item_ids": ["item-dev"],
        "verification": {
            "support_threshold": 0.65,
            "contradiction_threshold": 0.60,
        },
        "severity": {
            "high_contradiction_threshold": 0.75,
        },
        "provenance": {
            "prompt_sha256": prompt_hash,
            "evidence_sha256": evidence_hash,
            "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")

    response = ModelResponse("model-a", "item-test", "The result is supported.")
    JsonlResponseStore().save(responses, (response,))
    _write_jsonl(annotations, [{
        "schema_version": 1,
        "annotation_status": "adjudicated",
        "annotator_id": "adjudicator",
        "claim_review_status": "accepted",
        "model_name": "model-a",
        "dataset_item_id": "item-test",
        "claim_id": "claim-1",
        "claim_text": response.text,
        "claim_start": 0,
        "claim_end": len(response.text),
        "verification_label": "supported",
        "evidence_ids": ["evidence-test"],
        "severity": None,
        "rationale": "Supported by the trusted evidence.",
    }])
    return _paths(tmp_path)


def test_locked_test_blocks_missing_artifact(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing required locked-test artifact"):
        validate_locked_test(*paths)


def test_locked_test_returns_locked_thresholds(tmp_path):
    thresholds = validate_locked_test(*_valid_fixture(tmp_path))

    assert thresholds == {
        "support_threshold": 0.65,
        "contradiction_threshold": 0.60,
        "severity_contradiction_threshold": 0.75,
    }


def test_locked_test_rejects_benchmark_hash_mismatch(tmp_path):
    paths = _valid_fixture(tmp_path)
    prompts = paths[0]
    prompts.write_text(prompts.read_text(encoding="utf-8") + "changed", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen benchmark manifest"):
        validate_locked_test(*paths)


def test_locked_test_rejects_calibration_split_mismatch(tmp_path):
    paths = _valid_fixture(tmp_path)
    split = paths[3]
    split.write_text(json.dumps({
        "schema_version": 1,
        "prompt_sha256": "wrong",
        "evidence_sha256": "wrong",
        "item_splits": {"item-test": "test", "item-dev": "dev", "item-train": "train"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt hash does not match"):
        validate_locked_test(*paths)


def test_locked_test_rejects_dev_test_overlap(tmp_path):
    paths = _valid_fixture(tmp_path)
    calibration = paths[4]
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload["development_item_ids"] = ["item-test"]
    calibration.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Development and test item IDs overlap"):
        validate_locked_test(*paths)

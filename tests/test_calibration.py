import hashlib
import json
from dataclasses import asdict

import pytest

from hallucination_detector.calibration import (
    CalibrationObservation,
    SeverityCalibrationObservation,
    calibrate_severity_threshold,
    calibrate_verification_thresholds,
    create_topic_split,
    evaluate_severity,
)
from hallucination_detector.domain import Claim, ModelResponse, Severity, VerificationLabel
from hallucination_detector.generation import JsonlResponseStore
from scripts.calibrate_detector import calibrate


def test_verification_threshold_grid_search_selects_perfect_dev_cutoffs():
    observations = (
        CalibrationObservation(("model-a", "item-1", "claim-1"), 0.75, 0.1, VerificationLabel.SUPPORTED),
        CalibrationObservation(("model-a", "item-2", "claim-1"), 0.1, 0.75, VerificationLabel.CONTRADICTED),
        CalibrationObservation(("model-a", "item-3", "claim-1"), 0.4, 0.4, VerificationLabel.UNKNOWN),
    )

    result = calibrate_verification_thresholds(
        observations,
        candidate_thresholds=(0.6, 0.8),
    )

    assert result.support_threshold == 0.6
    assert result.contradiction_threshold == 0.6
    assert result.objective_score == 1.0
    assert result.accuracy == 1.0
    assert result.coverage == pytest.approx(2 / 3)
    assert result.candidates_evaluated == 4


def test_verification_threshold_tuning_validates_samples_and_candidates():
    sample = CalibrationObservation(
        ("model-a", "item-1", "claim-1"), 0.8, 0.1, VerificationLabel.SUPPORTED
    )
    with pytest.raises(ValueError, match="At least one development"):
        calibrate_verification_thresholds(())
    with pytest.raises(ValueError, match="unique claim keys"):
        calibrate_verification_thresholds((sample, sample))
    with pytest.raises(ValueError, match="candidate threshold"):
        calibrate_verification_thresholds((sample,), candidate_thresholds=(1.1,))


def test_severity_threshold_search_and_ordinal_metrics():
    observations = (
        SeverityCalibrationObservation(
            ("model-a", "item-1", "claim-1"),
            Claim("claim-1", "The finding was incorrect."),
            VerificationLabel.CONTRADICTED,
            0.9,
            Severity.SEVERE,
        ),
        SeverityCalibrationObservation(
            ("model-a", "item-2", "claim-1"),
            Claim("claim-1", "The report listed the wrong year."),
            VerificationLabel.CONTRADICTED,
            0.2,
            Severity.MODERATE,
        ),
    )

    calibrated = calibrate_severity_threshold(observations, candidate_thresholds=(0.5, 0.95))
    metrics = evaluate_severity(
        {
            observations[0].key: Severity.SEVERE,
            observations[1].key: Severity.MODERATE,
        },
        {
            observations[0].key: Severity.SEVERE,
            observations[1].key: Severity.MODERATE,
        },
    )

    assert calibrated.high_contradiction_threshold == 0.5
    assert calibrated.metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.quadratic_weighted_kappa == 1.0


def test_severity_metrics_exclude_supported_nulls_and_reject_missing_predictions():
    gold = {
        ("model-a", "item-1", "claim-1"): None,
        ("model-a", "item-1", "claim-2"): Severity.MILD,
    }
    with pytest.raises(ValueError, match="Missing predicted severity"):
        evaluate_severity(gold, {})
    with pytest.raises(ValueError, match="non-null gold severity"):
        evaluate_severity({("model-a", "item-1", "claim-1"): None}, {})


def test_topic_split_is_deterministic_and_keeps_prompt_variants_together():
    topic_by_item = {
        f"item-{topic}-{variant}": f"topic-{topic}"
        for topic in range(30)
        for variant in range(10)
    }

    first = create_topic_split(topic_by_item, seed=42)
    second = create_topic_split(topic_by_item, seed=42)
    topic_splits = {}
    for item_id, topic in topic_by_item.items():
        topic_splits.setdefault(topic, set()).add(first[item_id])

    assert first == second
    assert all(len(splits) == 1 for splits in topic_splits.values())
    assert {split: len({topic for topic, splits in topic_splits.items() if split in splits}) for split in ("train", "dev", "test")} == {
        "train": 18,
        "dev": 6,
        "test": 6,
    }


def test_topic_split_rejects_too_few_topics_and_invalid_fractions():
    with pytest.raises(ValueError, match="At least three topics"):
        create_topic_split({"item-1": "topic-a", "item-2": "topic-b"})
    with pytest.raises(ValueError, match="sum to less than 1"):
        create_topic_split(
            {f"item-{index}": f"topic-{index}" for index in range(3)},
            development_fraction=0.5,
            test_fraction=0.5,
        )


def test_calibration_pipeline_uses_only_dev_items_and_saves_provenance(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    evidence_path = tmp_path / "trusted_evidence.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    annotations_path = tmp_path / "adjudicated.jsonl"
    report_path = tmp_path / "evaluation-report.json"
    split_path = tmp_path / "topic-split.json"
    output_path = tmp_path / "calibration.json"

    prompts = []
    evidence = []
    responses = []
    annotations = []
    report_responses = []
    item_splits = {"item-dev": "dev", "item-test": "test", "item-train": "train"}
    topic_splits = {"topic-dev": "dev", "topic-test": "test", "topic-train": "train"}
    for item_id, split_name in item_splits.items():
        topic = f"topic-{split_name}"
        evidence_id = f"evidence-{split_name}"
        response = ModelResponse("model-a", item_id, "The year is 2020.")
        prompts.append(
            {
                "dataset_item_id": item_id,
                "topic": topic,
                "prompt": "What year is it?",
                "reference_evidence_ids": [evidence_id],
            }
        )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "topic": topic,
                "text": "The year is 2020.",
                "source": "Test source",
            }
        )
        responses.append(response)
        label = "contradicted" if split_name == "dev" else "supported"
        severity = "moderate" if split_name == "dev" else None
        annotations.append(
            {
                "schema_version": 1,
                "annotation_status": "adjudicated",
                "annotator_id": "adjudicator",
                "claim_review_status": "accepted",
                "model_name": response.model_name,
                "dataset_item_id": item_id,
                "claim_id": "claim-1",
                "claim_text": response.text,
                "claim_start": 0,
                "claim_end": len(response.text),
                "verification_label": label,
                "evidence_ids": [evidence_id],
                "severity": severity,
                "rationale": "Checked against the reference passage.",
            }
        )
        report_responses.append(
            {
                "response": asdict(response),
                "claims": [
                    {
                        "verification": {
                            "claim": {
                                "id": "claim-1",
                                "text": response.text,
                                "start": 0,
                                "end": len(response.text),
                            },
                            "support_score": 0.1 if split_name == "dev" else 0.9,
                            "contradiction_score": 0.75 if split_name == "dev" else 0.05,
                        },
                        "severity": {"severity": severity},
                        "retrieved_evidence": [{"evidence": {"id": evidence_id}}],
                    }
                ],
            }
        )

    def write_jsonl(path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    write_jsonl(prompts_path, prompts)
    write_jsonl(evidence_path, evidence)
    JsonlResponseStore().save(responses_path, tuple(responses))
    write_jsonl(annotations_path, annotations)
    report_path.write_text(
        json.dumps({"schema_version": 1, "responses": report_responses}),
        encoding="utf-8",
    )
    prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (tmp_path / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "healthcare-phase-1",
                "files": {
                    prompts_path.name: prompt_hash,
                    evidence_path.name: evidence_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompt_sha256": prompt_hash,
                "evidence_sha256": evidence_hash,
                "item_splits": item_splits,
                "topic_splits": topic_splits,
            }
        ),
        encoding="utf-8",
    )

    result = calibrate(
        report_path,
        responses_path,
        annotations_path,
        prompts_path,
        evidence_path,
        split_path,
        output_path,
        threshold_step=0.5,
    )

    assert result["development_item_ids"] == ["item-dev"]
    assert result["development_claim_count"] == 1
    assert result["verification"]["objective_score"] == 1.0
    assert result["severity"]["accuracy"] == 1.0
    assert result["retrieval_recall_at_k"]["1"] == 1.0
    assert output_path.exists()
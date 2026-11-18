import csv
import json

import pytest

from hallucination_detector.reporting import (
    build_research_summary,
    package_research_results,
    validate_release_report,
)


def _report():
    return {
        "schema_version": 1,
        "dataset_item_ids": ["item-1", "item-2"],
        "responses": [{"response": {}}, {"response": {}}],
        "model_metrics": [
            {
                "model_name": "model-a",
                "response_count": 2,
                "dataset_item_count": 2,
                "claim_count": 3,
                "supported_rate": 0.67,
                "contradicted_rate": 0.33,
                "unknown_rate": 0.0,
                "hallucination_rate": 0.33,
                "severity_weighted_error_rate": 0.12,
                "average_trust_score": 0.71,
                "verification_accuracy": 0.80,
                "verification_macro_f1": 0.75,
                "retrieval_recall_at_k": 0.90,
                "confidence_brier_score": 0.10,
                "confidence_ece": 0.08,
            }
        ],
        "paired_comparisons": [
            {
                "model_a": "model-a",
                "model_b": "model-b",
                "paired_item_count": 2,
                "supported_rate_difference": {
                    "mean_difference": 0.10,
                    "lower": 0.02,
                    "upper": 0.18,
                },
                "mcnemar_exact_p_value": 0.50,
                "model_a_error_only_items": 1,
                "model_b_error_only_items": 0,
            }
        ],
    }


def test_validate_release_report_requires_version_and_arrays():
    validate_release_report(_report())
    with pytest.raises(ValueError, match="schema_version"):
        validate_release_report({"schema_version": 2})
    with pytest.raises(ValueError, match="dataset_item_ids"):
        validate_release_report({"schema_version": 1, "dataset_item_ids": {}})


def test_build_research_summary_collapses_model_and_comparison_metrics():
    summary = build_research_summary(_report())

    assert summary["dataset_item_count"] == 2
    assert summary["model_count"] == 1
    assert summary["response_count"] == 2
    assert summary["claim_count"] == 3
    assert summary["models"][0]["model_name"] == "model-a"
    assert summary["paired_comparisons"][0]["model_b"] == "model-b"


def test_package_research_results_writes_csv_manifest_and_provenance(tmp_path):
    report_path = tmp_path / "test-report.json"
    split_path = tmp_path / "topic-split.json"
    calibration_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    split_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    calibration_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    output_dir = tmp_path / "release"

    manifest = package_research_results(
        report_path,
        output_dir,
        provenance_paths=(split_path, calibration_path),
    )

    assert (output_dir / "research_summary.json").exists()
    assert (output_dir / "model_metrics.csv").exists()
    assert (output_dir / "paired_comparisons.csv").exists()
    assert manifest["report"]["evaluation_report_sha256"]
    assert "topic-split.json_sha256" in manifest["report"]
    assert "calibration.json_sha256" in manifest["report"]

    with (output_dir / "model_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["model_name"] == "model-a"

    with (output_dir / "paired_comparisons.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["model_a"] == "model-a"
    assert float(rows[0]["supported_rate_difference_lower"]) == 0.02


def test_package_research_results_rejects_missing_report_and_existing_directory(tmp_path):
    report_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Missing evaluation report"):
        package_research_results(report_path, tmp_path / "release")

    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    output_dir = tmp_path / "release"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty release directory"):
        package_research_results(report_path, output_dir)


def test_package_research_results_rejects_missing_provenance(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing provenance artifact"):
        package_research_results(report_path, tmp_path / "release", provenance_paths=(tmp_path / "missing.json",))

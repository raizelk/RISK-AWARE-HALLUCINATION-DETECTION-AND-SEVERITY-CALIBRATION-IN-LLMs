"""Research-result packaging and provenance helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("Report must use schema_version 1.")
    for field in ("dataset_item_ids", "responses", "model_metrics", "paired_comparisons"):
        if not isinstance(report.get(field), list):
            raise ValueError(f"Report field {field!r} must be a JSON array.")


def build_research_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact, deterministic summary of an evaluation report."""
    validate_release_report(report)
    model_metrics = tuple(
        {
            "model_name": item.get("model_name"),
            "response_count": item.get("response_count"),
            "dataset_item_count": item.get("dataset_item_count"),
            "claim_count": item.get("claim_count"),
            "supported_rate": item.get("supported_rate"),
            "contradicted_rate": item.get("contradicted_rate"),
            "unknown_rate": item.get("unknown_rate"),
            "hallucination_rate": item.get("hallucination_rate"),
            "severity_weighted_error_rate": item.get("severity_weighted_error_rate"),
            "average_trust_score": item.get("average_trust_score"),
            "verification_accuracy": item.get("verification_accuracy"),
            "verification_macro_f1": item.get("verification_macro_f1"),
            "retrieval_recall_at_k": item.get("retrieval_recall_at_k"),
            "confidence_brier_score": item.get("confidence_brier_score"),
            "confidence_ece": item.get("confidence_ece"),
        }
        for item in report["model_metrics"]
        if isinstance(item, Mapping)
    )
    comparisons = tuple(
        {
            "model_a": item.get("model_a"),
            "model_b": item.get("model_b"),
            "paired_item_count": item.get("paired_item_count"),
            "supported_rate_difference": item.get("supported_rate_difference"),
            "mcnemar_exact_p_value": item.get("mcnemar_exact_p_value"),
            "model_a_error_only_items": item.get("model_a_error_only_items"),
            "model_b_error_only_items": item.get("model_b_error_only_items"),
        }
        for item in report["paired_comparisons"]
        if isinstance(item, Mapping)
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_item_ids": list(report["dataset_item_ids"]),
        "dataset_item_count": len(report["dataset_item_ids"]),
        "model_count": len(model_metrics),
        "response_count": sum(item["response_count"] or 0 for item in model_metrics),
        "claim_count": sum(item["claim_count"] or 0 for item in model_metrics),
        "models": list(model_metrics),
        "paired_comparisons": list(comparisons),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package_research_results(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    provenance_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Write a summary, model/comparison CSVs, and provenance manifest."""
    report_file = Path(report_path)
    destination = Path(output_dir)
    if not report_file.exists():
        raise FileNotFoundError(f"Missing evaluation report: {report_file}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty release directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid evaluation report JSON: {error.msg}") from error
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must contain a JSON object.")
    summary = build_research_summary(report)

    model_fields = (
        "model_name", "response_count", "dataset_item_count", "claim_count",
        "supported_rate", "contradicted_rate", "unknown_rate", "hallucination_rate",
        "severity_weighted_error_rate", "average_trust_score", "verification_accuracy",
        "verification_macro_f1", "retrieval_recall_at_k", "confidence_brier_score",
        "confidence_ece",
    )
    write_csv(destination / "model_metrics.csv", model_fields, summary["models"])

    comparison_fields = (
        "model_a", "model_b", "paired_item_count", "supported_rate_difference",
        "mcnemar_exact_p_value", "model_a_error_only_items", "model_b_error_only_items",
    )
    comparison_rows = []
    for comparison in summary["paired_comparisons"]:
        interval = comparison.pop("supported_rate_difference", None)
        row = dict(comparison)
        if isinstance(interval, Mapping):
            row["supported_rate_difference"] = interval.get("mean_difference")
            row["supported_rate_difference_lower"] = interval.get("lower")
            row["supported_rate_difference_upper"] = interval.get("upper")
        else:
            row["supported_rate_difference"] = None
        comparison_rows.append(row)
    comparison_fields = comparison_fields + (
        "supported_rate_difference_lower", "supported_rate_difference_upper"
    )
    write_csv(destination / "paired_comparisons.csv", comparison_fields, comparison_rows)

    summary_path = destination / "research_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    provenance: dict[str, str] = {
        "evaluation_report": str(report_file.resolve()),
        "evaluation_report_sha256": _sha256(report_file),
    }
    for raw_path in provenance_paths:
        artifact = Path(raw_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Missing provenance artifact: {artifact}")
        provenance[artifact.name] = str(artifact.resolve())
        provenance[f"{artifact.name}_sha256"] = _sha256(artifact)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": provenance,
        "files": {
            "research_summary.json": _sha256(summary_path),
            "model_metrics.csv": _sha256(destination / "model_metrics.csv"),
            "paired_comparisons.csv": _sha256(destination / "paired_comparisons.csv"),
        },
    }
    manifest_path = destination / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

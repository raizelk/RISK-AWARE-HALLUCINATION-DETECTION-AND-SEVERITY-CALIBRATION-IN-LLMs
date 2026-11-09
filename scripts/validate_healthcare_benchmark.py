"""Validate and optionally freeze the Phase 1 healthcare benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "healthcare"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"
EVIDENCE_PATH = DATA_DIR / "trusted_evidence.jsonl"
MANIFEST_PATH = DATA_DIR / "benchmark_manifest.json"

PROMPT_FIELDS = {
    "dataset_item_id",
    "domain",
    "topic",
    "task_type",
    "risk_level",
    "prompt",
    "reference_evidence_ids",
}
EVIDENCE_FIELDS = {
    "evidence_id",
    "domain",
    "topic",
    "text",
    "source",
    "source_url",
    "publication_date",
}
SOURCE_HOSTS = {
    "World Health Organization": "who.int",
    "Centers for Disease Control and Prevention": "cdc.gov",
    "U.S. Food and Drug Administration": "fda.gov",
    "National Heart, Lung, and Blood Institute": "nhlbi.nih.gov",
    "National Institute of Diabetes and Digestive and Kidney Diseases": "niddk.nih.gov",
}
GENERIC_PATHS = {"/health", "/health-topics", "/drugs", "/what-to-do"}


def load_jsonl(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return records, [f"Cannot read {path.relative_to(ROOT)}: {error}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            issues.append(f"{path.name}:{line_number}: blank lines are not allowed")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(f"{path.name}:{line_number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(value, dict):
            issues.append(f"{path.name}:{line_number}: each record must be a JSON object")
            continue
        records.append(value)
    return records, issues


def _check_fields(
    records: list[dict[str, object]], fields: set[str], path: Path, issues: list[str]
) -> None:
    for index, record in enumerate(records, start=1):
        missing = sorted(fields - record.keys())
        if missing:
            issues.append(f"{path.name}:{index}: missing fields: {', '.join(missing)}")


def validate() -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    prompts, issues = load_jsonl(PROMPTS_PATH)
    evidence, evidence_issues = load_jsonl(EVIDENCE_PATH)
    issues.extend(evidence_issues)
    _check_fields(prompts, PROMPT_FIELDS, PROMPTS_PATH, issues)
    _check_fields(evidence, EVIDENCE_FIELDS, EVIDENCE_PATH, issues)

    prompt_ids = [record.get("dataset_item_id") for record in prompts]
    evidence_ids = [record.get("evidence_id") for record in evidence]
    for label, values in (("prompt", prompt_ids), ("evidence", evidence_ids)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            issues.append(f"Duplicate {label} IDs: {duplicates}")

    evidence_by_id = {
        record["evidence_id"]: record
        for record in evidence
        if isinstance(record.get("evidence_id"), str)
    }
    topic_counts: Counter[str] = Counter()
    for index, record in enumerate(prompts, start=1):
        if record.get("domain") != "healthcare":
            issues.append(f"prompts.jsonl:{index}: domain must be 'healthcare'")
        if record.get("task_type") not in {"factual", "explanation", "advice"}:
            issues.append(f"prompts.jsonl:{index}: invalid task_type")
        if record.get("risk_level") not in {"moderate", "high"}:
            issues.append(f"prompts.jsonl:{index}: invalid risk_level")
        topic = record.get("topic")
        if isinstance(topic, str):
            topic_counts[topic] += 1
        references = record.get("reference_evidence_ids")
        if not isinstance(references, list) or not references:
            issues.append(f"prompts.jsonl:{index}: reference_evidence_ids must be a non-empty array")
            continue
        for evidence_id in references:
            linked = evidence_by_id.get(evidence_id)
            if linked is None:
                issues.append(f"prompts.jsonl:{index}: unknown evidence ID {evidence_id!r}")
            elif linked.get("topic") != topic:
                issues.append(f"prompts.jsonl:{index}: linked evidence topic does not match prompt topic")

    for index, record in enumerate(evidence, start=1):
        if record.get("domain") != "healthcare":
            issues.append(f"trusted_evidence.jsonl:{index}: domain must be 'healthcare'")
        source = record.get("source")
        url = record.get("source_url")
        parsed = urlparse(url) if isinstance(url, str) else None
        expected_host = SOURCE_HOSTS.get(source) if isinstance(source, str) else None
        hostname = parsed.hostname.lower() if parsed and parsed.hostname else ""
        if not parsed or parsed.scheme != "https" or not hostname:
            issues.append(f"trusted_evidence.jsonl:{index}: source_url must be an HTTPS URL")
        elif expected_host and not (hostname == expected_host or hostname.endswith(f".{expected_host}")):
            issues.append(f"trusted_evidence.jsonl:{index}: URL host {hostname!r} does not match {source!r}")
        if parsed and parsed.path.rstrip("/") in GENERIC_PATHS:
            issues.append(f"trusted_evidence.jsonl:{index}: source_url is a generic landing page, not a topic-specific citation")
        publication_date = record.get("publication_date")
        try:
            if not isinstance(publication_date, str):
                raise ValueError
            date.fromisoformat(publication_date)
        except ValueError:
            issues.append(f"trusted_evidence.jsonl:{index}: publication_date must be an ISO date")

    if len(prompts) != 300:
        issues.append(f"Expected 300 prompts for the frozen Phase 1 set; found {len(prompts)}")
    if len(evidence) != 30:
        issues.append(f"Expected 30 evidence records for the frozen Phase 1 set; found {len(evidence)}")
    if any(count != 10 for count in topic_counts.values()):
        issues.append("Each topic must have exactly 10 prompts")
    return issues, prompts, evidence


def write_manifest(prompts: list[dict[str, object]], evidence: list[dict[str, object]]) -> None:
    manifest = {
        "schema_version": 1,
        "dataset": "healthcare-phase-1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(prompts),
        "evidence_count": len(evidence),
        "topic_count": len({record.get("topic") for record in prompts}),
        "files": {
            PROMPTS_PATH.name: hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
            EVIDENCE_PATH.name: hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest(),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="write a hash manifest only when all validation checks pass",
    )
    args = parser.parse_args()
    issues, prompts, evidence = validate()
    if issues:
        print(f"BLOCKED: {len(issues)} validation issue(s); benchmark was not frozen.")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"PASS: {len(prompts)} prompts, {len(evidence)} evidence records.")
    if args.freeze:
        write_manifest(prompts, evidence)
        print(f"Frozen benchmark manifest written to {MANIFEST_PATH.relative_to(ROOT)}")
    else:
        print("Run with --freeze to write the benchmark hash manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
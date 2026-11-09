"""Export prediction-free claim annotation JSONL from saved model responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.baseline import extract_claims
from hallucination_detector.generation import JsonlResponseStore
from hallucination_detector.runner import load_benchmark


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path.name} line {line_number}.") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} line {line_number} must be a JSON object.")
            records.append(value)
    return records


def export_template(
    responses_path: Path,
    prompts_path: Path,
    evidence_path: Path,
    output_path: Path,
    *,
    annotator_id: str,
    allow_unfrozen_pilot: bool = False,
) -> int:
    if not annotator_id.strip():
        raise ValueError("annotator_id must not be empty.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite annotation file {output_path}.")

    benchmark = load_benchmark(prompts_path, evidence_path)
    if not benchmark.is_frozen and not allow_unfrozen_pilot:
        raise ValueError(
            "The benchmark has no valid freeze manifest. Fix and freeze it first, "
            "or pass --allow-unfrozen-pilot for pilot-only annotation."
        )

    prompts = {record["dataset_item_id"]: record for record in _read_jsonl(prompts_path)}
    evidence = {record["evidence_id"]: record for record in _read_jsonl(evidence_path)}
    responses = JsonlResponseStore().load(responses_path)
    if not responses:
        raise ValueError("Response JSONL contains no model responses.")

    response_keys = [(item.model_name, item.dataset_item_id) for item in responses]
    if len(response_keys) != len(set(response_keys)):
        raise ValueError("Response JSONL contains duplicate model/item pairs.")
    models = tuple(dict.fromkeys(item.model_name for item in responses))
    items = tuple(dict.fromkeys(item.dataset_item_id for item in responses))
    expected_pairs = {(model, item) for model in models for item in items}
    if set(response_keys) != expected_pairs:
        raise ValueError("Responses must contain a complete model-by-item matrix before annotation.")

    rows: list[dict[str, object]] = []
    response_without_claims = 0
    for response in responses:
        prompt_record = prompts.get(response.dataset_item_id)
        if prompt_record is None:
            raise ValueError(f"Unknown dataset item {response.dataset_item_id!r} in responses.")
        candidates = []
        for evidence_id in prompt_record.get("reference_evidence_ids", []):
            record = evidence[evidence_id]
            candidates.append(
                {
                    "evidence_id": evidence_id,
                    "text": record["text"],
                    "source": record["source"],
                    "source_url": record["source_url"],
                    "publication_date": record["publication_date"],
                }
            )
        claims = extract_claims(response.text)
        if not claims:
            response_without_claims += 1
        for claim in claims:
            rows.append(
                {
                    "schema_version": 1,
                    "annotation_status": "independent",
                    "annotator_id": annotator_id,
                    "claim_review_status": "pending",
                    "model_name": response.model_name,
                    "dataset_item_id": response.dataset_item_id,
                    "prompt": prompt_record["prompt"],
                    "domain": prompt_record.get("domain"),
                    "topic": prompt_record.get("topic"),
                    "task_type": prompt_record.get("task_type"),
                    "risk_level": prompt_record.get("risk_level"),
                    "response_text": response.text,
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "claim_start": claim.start,
                    "claim_end": claim.end,
                    "verification_label": None,
                    "evidence_ids": [],
                    "severity": None,
                    "rationale": "",
                    "evidence_candidates": candidates,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    if not benchmark.is_frozen:
        print("WARNING: this annotation template is for pilot use only; the benchmark is unfrozen.")
    print(
        f"Exported {len(rows)} claim rows from {len(responses)} responses to {output_path}; "
        f"{response_without_claims} responses had no extracted claims."
    )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=ROOT / "data/pilot-responses.jsonl")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/healthcare/prompts.jsonl")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument(
        "--allow-unfrozen-pilot",
        action="store_true",
        help="allow worksheet export from the current unfrozen benchmark for pilot use only",
    )
    args = parser.parse_args(argv)
    try:
        export_template(
            args.responses,
            args.prompts,
            args.evidence,
            args.output,
            annotator_id=args.annotator_id,
            allow_unfrozen_pilot=args.allow_unfrozen_pilot,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
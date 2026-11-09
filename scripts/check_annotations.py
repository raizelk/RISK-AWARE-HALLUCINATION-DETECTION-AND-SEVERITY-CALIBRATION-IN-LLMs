"""Compare independent annotations or validate adjudicated gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.annotations import (
    compare_independent_annotations,
    load_adjudicated_annotations,
)
from hallucination_detector.domain import Severity, VerificationLabel
from hallucination_detector.generation import JsonlResponseStore


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=ROOT / "data/pilot-responses.jsonl")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adjudicated", type=Path, help="validate a final adjudicated labels file")
    group.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("ANNOTATOR_A", "ANNOTATOR_B"),
        help="compare two independent annotation files",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require one adjudicated row for every extracted claim",
    )
    args = parser.parse_args(argv)

    try:
        responses = JsonlResponseStore().load(args.responses)
        evidence_ids = _read_evidence_ids(args.evidence)
        if args.compare:
            agreement = compare_independent_annotations(
                args.compare[0], args.compare[1], responses, evidence_ids
            )
            kappa = "undefined" if agreement.cohen_kappa is None else f"{agreement.cohen_kappa:.4f}"
            print(f"Compared claims: {agreement.compared_claims}")
            print(f"Exact label agreement: {agreement.exact_agreement:.4f}")
            print(f"Cohen's kappa: {kappa}")
            print(f"Disagreements: {len(agreement.disagreement_keys)}")
            for key in agreement.disagreement_keys:
                print(f"- {key[0]} / {key[1]} / {key[2]}")
        else:
            gold = load_adjudicated_annotations(
                args.adjudicated,
                responses,
                evidence_ids,
                require_complete=args.require_complete,
            )
            labels = Counter(gold.labels.values())
            severities = Counter(
                severity for severity in gold.severity.values() if severity is not None
            )
            print(f"Validated adjudicated claims: {len(gold.annotations)}")
            print(
                "Verification labels: "
                + ", ".join(f"{label.value}={labels[label]}" for label in VerificationLabel)
            )
            print(
                "Severity labels: "
                + ", ".join(f"{severity.value}={severities[severity]}" for severity in Severity)
            )
            print(f"Claims with evidence IDs: {len(gold.evidence_ids)}")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
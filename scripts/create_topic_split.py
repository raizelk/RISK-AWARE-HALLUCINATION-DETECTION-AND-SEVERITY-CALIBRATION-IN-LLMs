"""Create a deterministic topic-disjoint train/dev/test manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.calibration import create_topic_split
from hallucination_detector.runner import load_benchmark, write_json_atomic


def _read_prompt_topics(path: Path) -> dict[str, str]:
    topic_by_item: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid prompt JSON on line {line_number}.") from error
            if not isinstance(record, dict):
                raise ValueError(f"Prompt line {line_number} must be a JSON object.")
            item_id, topic = record.get("dataset_item_id"), record.get("topic")
            if not isinstance(item_id, str) or not isinstance(topic, str):
                raise ValueError(f"Prompt line {line_number} needs dataset_item_id and topic.")
            if item_id in topic_by_item:
                raise ValueError(f"Duplicate dataset item ID {item_id!r}.")
            topic_by_item[item_id] = topic
    return topic_by_item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/healthcare/prompts.jsonl")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/healthcare/topic-split.json"
    )
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"Refusing to overwrite existing split manifest: {args.output}")
    try:
        benchmark = load_benchmark(args.prompts, args.evidence)
        if not benchmark.is_frozen:
            raise ValueError("Freeze the reviewed benchmark before assigning train/dev/test topics.")
        topic_by_item = _read_prompt_topics(args.prompts)
        item_splits = create_topic_split(
            topic_by_item,
            development_fraction=args.development_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        topic_splits: dict[str, str] = {}
        for item_id, topic in topic_by_item.items():
            split = item_splits[item_id]
            previous = topic_splits.setdefault(topic, split)
            if previous != split:
                raise ValueError(f"Topic {topic!r} was assigned across multiple splits.")
        counts = {
            split: sum(value == split for value in topic_splits.values())
            for split in ("train", "dev", "test")
        }
        write_json_atomic(
            args.output,
            {
                "schema_version": 1,
                "dataset": "healthcare-phase-1",
                "seed": args.seed,
                "development_fraction": args.development_fraction,
                "test_fraction": args.test_fraction,
                "prompt_sha256": benchmark.prompt_sha256,
                "evidence_sha256": benchmark.evidence_sha256,
                "topic_splits": topic_splits,
                "item_splits": item_splits,
                "topic_counts": counts,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote topic-disjoint split manifest: {args.output}")
    print(f"Topics by split: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Check code health and report whether the final research release has required data artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RELEASE_ARTIFACTS = (
    "data/healthcare/benchmark_manifest.json",
    "data/healthcare/topic-split.json",
    "data/calibration-dev.json",
    "data/responses.jsonl",
    "data/annotations/adjudicated.jsonl",
    "data/test-evaluation-report.json",
)


def _pytest() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return False, str(error)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="run the complete pytest suite")
    args = parser.parse_args(argv)

    if args.run_tests:
        passed, output = _pytest()
        print("TESTS PASS" if passed else "TESTS FAIL")
        if output:
            print(output)
        if not passed:
            return 1

    missing = [name for name in REQUIRED_RELEASE_ARTIFACTS if not (ROOT / name).exists()]
    present = [name for name in REQUIRED_RELEASE_ARTIFACTS if (ROOT / name).exists()]
    print("RELEASE ARTIFACT STATUS")
    for name in present:
        print(f"- PRESENT: {name}")
    for name in missing:
        print(f"- MISSING: {name}")

    if missing:
        print("RELEASE STATUS: BLOCKED by missing experiment artifacts.")
        return 2
    print("RELEASE STATUS: READY for packaging and dashboard review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

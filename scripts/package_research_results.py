"""Package a validated evaluation report for reproducible result release."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.reporting import package_research_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        action="append",
        default=[],
        help="artifact to hash and include in the release manifest; repeat as needed",
    )
    args = parser.parse_args(argv)
    try:
        manifest = package_research_results(
            args.report,
            args.output_dir,
            provenance_paths=args.provenance,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    manifest_path = args.output_dir / "release_manifest.json"
    release_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"Packaged research results: {args.output_dir}")
    print(f"Release manifest: {manifest_path}")
    print(f"Release artifact SHA-256: {release_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

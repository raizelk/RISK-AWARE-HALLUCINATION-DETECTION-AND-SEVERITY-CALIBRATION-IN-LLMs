"""Probe configured live providers before starting benchmark generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import live_adapters


def main() -> int:
    import os

    from hallucination_detector.generation import PromptRecord

    configurations = (
        (
            "openrouter-hy3",
            "OPENROUTER_API_KEY",
            os.getenv("OPENROUTER_MODEL", "tencent/hy3:free"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        (
            "groq",
            "GROQ_API_KEY",
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        ),
        (
            "orcarouter-deepseek",
            "ORCAROUTER_API_KEY",
            os.getenv("ORCAROUTER_MODEL", "deepseek/deepseek-v4.1-flash-free"),
            os.getenv("ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1"),
        ),
    )

    passed = 0
    failed = False
    for provider, key_name, model, base_url in configurations:
        key = os.getenv(key_name)
        if not key:
            print(f"{provider}: BLOCKED missing {key_name}; HTTP not called")
            failed = True
            continue
        adapter = live_adapters.OpenAICompatibleAdapter(
            model_name=model,
            api_key=key,
            base_url=base_url,
            provider=provider,
            temperature=0.0,
            timeout=live_adapters._timeout(),
            metadata={"provider": provider, "model_version": model},
        )
        try:
            result = adapter.generate(PromptRecord(f"probe-{provider}", "Reply with exactly one word: ready."))
            print(f"{provider}: PASS success=true response={result.text[:80]!r}")
            passed += 1
        except Exception as error:
            print(f"{provider}: FAIL {error}")
            failed = True

    if passed == 0:
        print("PROVIDER PREFLIGHT FAILED: no configured provider passed.", file=sys.stderr)
        return 1
    print(f"PROVIDER PREFLIGHT PASSED with {passed} provider(s); optional provider failures do not block a single-provider pilot.")
    if failed:
        print("NOTE: unavailable providers will be skipped; use build_deepseek_adapter for this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

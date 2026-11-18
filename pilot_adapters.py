"""Deterministic offline adapters for pipeline and checkpoint testing.

These adapters are not research models and must not be used for final results.
Their responses come from the prompt text and are intended only to exercise
response generation, resume behavior, and split-aware evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from hallucination_detector.generation import LLMAdapter, PromptRecord, StaticLLMAdapter, GeneratedText

_ROOT = Path(__file__).resolve().parent
_PROMPTS = _ROOT / "data" / "healthcare" / "prompts.jsonl"


def _load_prompt_text() -> dict[str, str]:
    prompts: dict[str, str] = {}
    with _PROMPTS.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            prompts[record["dataset_item_id"]] = record["prompt"]
    return prompts


_PROMPT_TEXT = _load_prompt_text()


def build_adapters() -> tuple[LLMAdapter, ...]:
    """Return three named offline adapters for pilot pipeline testing only."""
    responses_a = {
        item_id: GeneratedText(
            f"For {prompt.lower().rstrip('?')}, consult authoritative healthcare guidance "
            "and a qualified clinician for individual medical decisions.",
            0.55,
        )
        for item_id, prompt in _PROMPT_TEXT.items()
    }
    responses_b = {
        item_id: GeneratedText(
            f"General evidence-based guidance for {prompt.lower().rstrip('?')}: "
            "seek professional medical advice when symptoms are uncertain or urgent.",
            0.50,
        )
        for item_id, prompt in _PROMPT_TEXT.items()
    }
    responses_c = {
        item_id: GeneratedText(
            f"A cautious summary of {prompt.lower().rstrip('?')} should be based on "
            "current trusted evidence and should not replace a clinician's diagnosis.",
            0.60,
        )
        for item_id, prompt in _PROMPT_TEXT.items()
    }

    metadata = {
        "provider": "offline",
        "model_version": "pilot-deterministic-v1",
        "research_use": False,
    }
    model_a = StaticLLMAdapter("pilot-model-a", responses_a, metadata)
    model_b = StaticLLMAdapter("pilot-model-b", responses_b, metadata)
    model_c = StaticLLMAdapter("pilot-model-c", responses_c, metadata)
    return model_a, model_b, model_c

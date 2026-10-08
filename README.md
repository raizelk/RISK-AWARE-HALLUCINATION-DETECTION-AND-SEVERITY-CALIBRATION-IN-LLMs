# Risk-Aware Hallucination Detection and Severity Calibration in LLMs

This project develops an evidence-grounded framework for detecting, explaining,
and scoring hallucinations in responses from three LLMs on a common dataset.

## Development roadmap

1. Dataset and ground-truth evidence/claims
2. Response generation for three model adapters
3. Atomic claim extraction
4. BM25 and dense evidence retrieval
5. Similarity and NLI verification
6. Severity classification
7. Claim-level explanations and response highlighting
8. Multi-signal trust scoring
9. Three-model evaluation
10. Dashboard and research results

## Quick start

The first vertical slice is dependency-free and can be run with Python 3.11+:

```powershell
python -m pytest
python -m hallucination_detector.demo
```

The package is laid out so that retrieval, verification, severity, explainability,
and scoring implementations can be added independently under `src/`.

## Project layout

```text
src/hallucination_detector/
	domain.py       # typed records shared by every sprint
	baseline.py     # deterministic first-pass claim/evidence pipeline
	generation.py   # provider-neutral three-model response generation and JSONL storage
	demo.py         # small local example
tests/
	test_baseline.py
	test_generation.py
```

## Sprint 2: response generation

Provider integrations implement `LLMAdapter` and return `GeneratedText`. The
generator preserves dataset/model order and emits `ModelResponse` records.
Responses can be stored as reproducible JSONL without requiring a provider SDK:

```python
from hallucination_detector.generation import (
    GeneratedText, JsonlResponseStore, PromptRecord, ResponseGenerator,
    StaticLLMAdapter,
)

records = (PromptRecord("item-1", "What is the capital of Australia?"),)
generator = ResponseGenerator((
    StaticLLMAdapter("model-a", {"item-1": GeneratedText("Canberra.")}),
    StaticLLMAdapter("model-b", {"item-1": GeneratedText("Sydney.")}),
    StaticLLMAdapter("model-c", {"item-1": GeneratedText("Canberra.")}),
))
responses = generator.generate(records)
JsonlResponseStore().save("data/responses.jsonl", responses)
```
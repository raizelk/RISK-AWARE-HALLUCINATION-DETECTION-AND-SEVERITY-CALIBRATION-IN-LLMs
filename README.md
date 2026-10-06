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
	demo.py         # small local example
tests/
	test_baseline.py
```
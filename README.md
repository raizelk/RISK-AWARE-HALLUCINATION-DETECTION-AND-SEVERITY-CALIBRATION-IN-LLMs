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

## Stage 4: hybrid evidence retrieval

`HybridEvidenceRetriever` ranks evidence passages with BM25 and dense embedding
similarity. BM25 is implemented locally; dense vectors use the optional Sentence
Transformers integration. Install the retrieval extra before using the real
embedding model (the model is downloaded and cached on first use):

```powershell
pip install -e ".[retrieval]"
```

Example, after obtaining a `Claim` and trusted `Evidence` records:

```python
from hallucination_detector.domain import Claim, Evidence
from hallucination_detector.retrieval import (
	HybridEvidenceRetriever,
	SentenceTransformerEmbedder,
)

evidence = (
	Evidence("source-1", "Canberra is the capital city of Australia.", "reference"),
	Evidence("source-2", "Sydney is the capital of New South Wales.", "reference"),
)
retriever = HybridEvidenceRetriever(evidence, SentenceTransformerEmbedder())
results = retriever.retrieve(Claim("claim-1", "Canberra is Australia's capital."), top_k=3)

for result in results:
	print(result.rank, result.evidence.id, result.bm25_score,
		  result.embedding_score, result.combined_score)
```

`bm25_score` is the raw BM25 value; `embedding_score` is cosine similarity
mapped to `[0, 1]`; and `combined_score` uses the configured weights (0.5 each
by default). Use `embedding_weight=0, bm25_weight=1` for BM25-only retrieval
without the optional package or model download. Reuse one retriever instance
across claims so indexed evidence embeddings are computed only once.

The package keeps retrieval, verification, severity, explainability, and scoring
as independently testable components under `src/`.

Evaluate retrieval separately using the gold evidence IDs for each claim:

```python
from hallucination_detector.retrieval_metrics import (
	mean_reciprocal_rank,
	precision_at_k,
	recall_at_k,
)

ranked_ids = [item.evidence.id for item in results]
gold_ids = {"source-1"}
print(recall_at_k(ranked_ids, gold_ids, k=5))
print(precision_at_k(ranked_ids, gold_ids, k=5))
print(mean_reciprocal_rank([ranked_ids], [gold_ids]))
```

## Stage 5: semantic and NLI verification

Stage 5 evaluates each retrieved evidence passage as the NLI premise and the
claim as the hypothesis. Install the optional verification dependencies; the
first use downloads/caches the embedding and NLI models:

```powershell
pip install -e ".[verification]"
```

```python
from hallucination_detector.domain import Claim, Evidence
from hallucination_detector.retrieval import (
	HybridEvidenceRetriever,
	SentenceTransformerEmbedder,
)
from hallucination_detector.verification import (
	EmbeddingSimilarityScorer,
	HuggingFaceNLIModel,
	verify_claim_with_nli,
)

embedder = SentenceTransformerEmbedder()
corpus = (
	Evidence("source-1", "Canberra is the capital city of Australia.", "reference"),
	Evidence("source-2", "Sydney is the capital of New South Wales.", "reference"),
)
claim = Claim("claim-1", "Canberra is the capital of Australia.")
retriever = HybridEvidenceRetriever(corpus, embedder)
retrieved = retriever.retrieve(claim, top_k=5)

result = verify_claim_with_nli(
	claim,
	retrieved,
	HuggingFaceNLIModel(),
	EmbeddingSimilarityScorer(embedder),
)
print(result.label, result.support_score, result.contradiction_score)
for item in result.evidence_scores:
	print(item.evidence.id, item.similarity, item.entailment_probability,
		  item.contradiction_probability, item.support_score, item.contradiction_score)
```

`verify_claim_with_nli()` returns `SUPPORTED`, `CONTRADICTED`, or `UNKNOWN`,
plus similarity, retrieval, entailment, contradiction, and neutral signals for
each candidate. Its initial adjusted scores are `NLI probability × retrieval
score × semantic similarity`; the default decision thresholds are provisional
and should be tuned on a manually labeled validation set. Missing or
inconclusive evidence remains `UNKNOWN`, not `CONTRADICTED`. The model adapter
checks label names and requires an explicit mapping if a model exposes
ambiguous labels such as `LABEL_0`.

## Stage 6: severity classification

`SeverityClassifier` converts a `VerificationResult` into a `SeverityAssessment`.
Supported claims receive no error severity. Unknown claims start at mild;
contradicted claims start at moderate. A strong contradiction score or high
response centrality can raise severity. A central claim in a detected
high-impact domain is at least severe, and unsupported actionable advice in
medical, safety, legal, or financial contexts is critical.

The classifier returns its rationale and detected risk domains with the
assessment. Centrality is caller-supplied on a `[0, 1]` scale; if unavailable,
it defaults to `0`. Its thresholds and lexical domain/action indicators are
transparent starting rules, not calibrated probabilities, and should be
validated against a labeled severity dataset before drawing research
conclusions.

```python
from hallucination_detector.severity import SeverityClassifier

assessment = SeverityClassifier().classify(
    verification_result,
    centrality=0.8,
)
print(assessment.severity, assessment.rationale)
```

## Stage 7: claim-level explanations and highlighting

`explain_response()` connects verified claims to their exact response spans,
retrieved evidence, verification reasons, and optional Stage 6 severity
assessments. It returns structured per-claim details and HTML with escaped
response text and `<mark>` tags around each verified claim. Claim offsets must
match the original response exactly; invalid or overlapping spans and
mismatched severity assessments are rejected.

```python
from hallucination_detector.explainability import explain_response

explanation = explain_response(
    original_response,
    (verification_result,),
    (severity_assessment,),
)
print(explanation.highlighted_response_html)
for item in explanation.claims:
    print(item.claim.text, item.verification_label, item.severity)
    print(item.evidence, item.verification_reason, item.severity_reason)
```

## Stage 8: multi-signal trust scoring

`TrustScorer` combines claim factuality, semantic alignment, response
consistency, and confidence calibration into a bounded score in `[0, 1]`.
Severity assessments add risk-aware penalties: mild, moderate, severe, and
critical errors contribute progressively larger penalties. The scorer returns
all component values and an auditable rationale rather than only one opaque
number.

```python
from hallucination_detector.trust import TrustScorer

trust = TrustScorer().score(
    verifications,
    severity_assessments=severity_assessments,
    consistency_score=0.85,
    confidences=(0.92, 0.40, 0.81),
)
print(trust.score, trust.factual_score, trust.severity_penalty)
```

`consistency_score` is supplied by a repeated or paraphrased-response
evaluation, while confidence calibration compares each supplied confidence
with the verified claim outcome. If confidence values are omitted, calibration
is reported as unavailable instead of being treated as perfect.

## Project layout

```text
src/hallucination_detector/
	domain.py       # typed records shared by every sprint
	baseline.py     # deterministic first-pass claim/evidence pipeline
	retrieval.py    # Stage 4 BM25 + dense hybrid evidence retrieval
	retrieval_metrics.py # Recall@K, Precision@K, and reciprocal-rank metrics
	verification.py # Stage 5 semantic similarity + NLI verification
	severity.py     # Stage 6 risk-aware claim severity assessment
	explainability.py # Stage 7 claim explanations and response highlighting
	trust.py        # Stage 8 multi-signal, severity-aware trust scoring
	generation.py   # provider-neutral three-model response generation and JSONL storage
	demo.py         # small local example
tests/
	test_baseline.py
	test_generation.py
	test_severity.py
	test_explainability.py
	test_trust.py
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
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

## Benchmark dataset

The benchmark is evaluated in separate phases. **Phase 1 is healthcare-only**
and contains 300 prompts in `data/healthcare/prompts.jsonl`. Finance remains
outside this phase and can be added later as `data/finance/` without changing
the healthcare evaluation set. The healthcare reference passages are in
`data/healthcare/trusted_evidence.jsonl`.

The original mixed six-healthcare/six-finance starter fixture remains in
`data/prompts.jsonl` and `data/trusted_evidence.jsonl` for backward
compatibility. Do not use it for the healthcare-only experiment.

Each prompt record uses this format:

```json
{
  "dataset_item_id": "healthcare-001",
  "domain": "healthcare",
  "task_type": "factual",
  "risk_level": "high",
  "prompt": "What is the recommended first-line treatment for ...?",
  "reference_evidence_ids": ["healthcare-evidence-001"]
}
```

Required prompt fields are:

- `dataset_item_id`: stable unique identifier
- `domain`: `healthcare` for the current phase
- `task_type`: `factual`, `explanation`, or `advice`
- `risk_level`: `moderate` or `high`
- `prompt`: the user question sent unchanged to every model
- `reference_evidence_ids`: one or more trusted evidence IDs

Each trusted evidence record uses this format:

```json
{
  "evidence_id": "healthcare-evidence-001",
  "domain": "healthcare",
  "text": "Reference passage used to verify model claims.",
  "source": "World Health Organization",
  "source_url": "https://www.who.int/...",
  "publication_date": "2023-09-15"
}
```

The evidence `text` is the content indexed by the retriever. The source
metadata makes each verification result traceable. These are starter records
for pipeline development; research results require source review, exact URL
verification, version locking, and independent human annotation of generated
claims.

### How `trusted_evidence.jsonl` is created

Trusted evidence is created independently from model outputs:

1. Define a healthcare topic and the factual claims the benchmark should test.
2. Locate an authoritative public source, preferably a government, public
   health, or professional medical organization.
3. Extract a short, self-contained passage that directly supports or refutes
   claims about that topic. Do not write the passage from an LLM response.
4. Record a stable `evidence_id`, domain, topic, passage text, source,
   canonical URL, and publication or review date.
5. Link each prompt to one or more evidence IDs using
   `reference_evidence_ids`.
6. Review the passage and URL manually, record the source version/date, and
   have an independent annotator verify the link before using the data for
   research metrics.

The current healthcare file has 30 topic-level reference records. Ten
question formulations are generated for each topic, producing 300 prompts.
The generated records are a reproducible starter benchmark, not medical
advice; the source URLs and wording must be reviewed before publication.

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

## Stage 9: three-model evaluation

`ModelComparisonEvaluator` applies the same claim extraction, evidence
retrieval, NLI verification, severity policy, and trust scorer to every model
response. Supply one response for every configured model/dataset-item pair;
incomplete matrices and duplicate responses are rejected.

```python
from hallucination_detector.evaluation import ModelComparisonEvaluator
from hallucination_detector.generation import JsonlResponseStore
from hallucination_detector.retrieval import HybridEvidenceRetriever
from hallucination_detector.verification import (
	EmbeddingSimilarityScorer,
	HuggingFaceNLIModel,
)

# Use the same evidence corpus and detector configuration for every model.
retriever = HybridEvidenceRetriever(corpus, embedder)
nli_model = HuggingFaceNLIModel()
similarity_scorer = EmbeddingSimilarityScorer(embedder)
responses = JsonlResponseStore().load("data/responses.jsonl")
evaluator = ModelComparisonEvaluator(
	model_names=("model-a", "model-b", "model-c"),
	dataset_item_ids=("item-1", "item-2"),
	retriever=retriever,
	nli_model=nli_model,
	similarity_scorer=similarity_scorer,
	top_k=5,
	bootstrap_repetitions=2000,
	random_seed=42,
)

# Optional consistency scores must come from repeated/paraphrased runs.
report = evaluator.evaluate(
	responses,
	consistency_scores={
		("model-a", "item-1"): 0.9,
		("model-a", "item-2"): 0.8,
		("model-b", "item-1"): 0.85,
		("model-b", "item-2"): 0.9,
		("model-c", "item-1"): 0.88,
		("model-c", "item-2"): 0.92,
	},
)

for metrics in report.model_metrics:
	print(
		metrics.model_name,
		metrics.supported_rate,
		metrics.hallucination_rate,
		metrics.unknown_rate,
		metrics.severity_counts,
		metrics.average_trust_score,
	)

for comparison in report.paired_comparisons:
	print(comparison.model_a, comparison.model_b,
		  comparison.supported_rate_difference,
		  comparison.mcnemar_exact_p_value)
```

For supervised verification metrics, pass `gold_labels` as a mapping from
`(model_name, dataset_item_id, claim_id)` to a `VerificationLabel`. For retrieval
Recall@K, pass `gold_evidence_ids` using the same keys and a collection of
relevant evidence IDs. These labels must come from independent human annotation
or another trusted gold dataset, not from the model outputs being evaluated.

`hallucination_rate` here means the share of claims labeled **contradicted**;
unknown claims have their own rate and are not counted as proven false.
`verification_accuracy`, macro-F1 and per-class metrics require `gold_labels`;
retrieval Recall@K requires `gold_evidence_ids`. Response-confidence Brier
score and ECE are reported only when a response has confidence and all its
claims have gold labels. Trust scores are reported only for responses with an
externally measured consistency score; consistency is not assumed to be
perfect. Pairwise comparisons use paired item-level supported rates with a
deterministic bootstrap confidence interval and an exact McNemar test for
whether either model produced a contradicted claim on an item. These statistics
are descriptive and require sufficient labeled benchmark data before drawing
research conclusions.

## Stage 10: dashboard and research results

Save the complete Stage 9 report, including claim-level verification and
retrieved evidence, then install and launch the optional interactive dashboard:

```python
report.save_json("data/evaluation_report.json")
```

```powershell
pip install -e ".[dashboard]"
streamlit run src/hallucination_detector/dashboard.py
```

The dashboard also accepts an uploaded Stage 9 JSON report. It includes an
overview of outcome rates, model comparison tables and paired statistics,
severity distributions and critical-claim review, claim-level explanations,
exact-offset highlighting in original responses, retrieved evidence with
BM25/embedding/NLI signals, and trust-score components.
Model and dataset-item filters drive the claim, risk, evidence, and trust
inspection views. Claim details can be downloaded as CSV and the full report as
JSON. The dashboard visualizes saved, versioned evaluation results; it does
not regenerate model outputs or rerun verification.

## Project layout

```text
src/hallucination_detector/
	domain.py       # typed records shared by every sprint
	baseline.py     # deterministic first-pass claim/evidence pipeline
	retrieval.py    # Stage 4 BM25 + dense hybrid evidence retrieval
	retrieval_metrics.py # Recall@K, Precision@K, and reciprocal-rank metrics
	verification.py # Stage 5 semantic similarity + NLI verification
	evaluation.py   # Stage 9 common-pipeline model comparison and paired metrics
	dashboard.py    # Stage 10 interactive research dashboard
	severity.py     # Stage 6 risk-aware claim severity assessment
	explainability.py # Stage 7 claim explanations and response highlighting
	trust.py        # Stage 8 multi-signal, severity-aware trust scoring
	generation.py   # provider-neutral three-model response generation and JSONL storage
	demo.py         # small local example
data/
	healthcare/
		prompts.jsonl          # 300 healthcare prompts for Phase 1
		trusted_evidence.jsonl # 30 healthcare reference passages
tests/
	test_baseline.py
	test_generation.py
	test_retrieval.py
	test_verification.py
	test_evaluation.py
	test_dashboard.py
	test_severity.py
	test_explainability.py
	test_trust.py
```

## Sprint 2: response generation

Provider integrations implement `LLMAdapter` and return `GeneratedText`. The
generator preserves dataset/model order and emits `ModelResponse` records.
Responses can be stored as reproducible JSONL without requiring a provider SDK:

### Pilot runner

The runner loads the healthcare JSONL files, checks evidence references and any
benchmark freeze manifest, and checkpoints each model response. It defaults to
five prompts and writes separate pilot output files. Provider adapters remain
plugin-based because the project does not select a model provider. A factory
module must expose a zero-argument function returning adapters that implement
`model_name` and `generate(PromptRecord) -> GeneratedText`; adapters may expose
a JSON-serializable `metadata` mapping for provider, model version, and
generation parameters. Do not put credentials in adapter metadata.

Install the optional retrieval and verification dependencies, then run a pilot:

```powershell
pip install -e ".[retrieval,verification]"
python scripts/run_real_evaluation.py `
	--adapter-factory my_adapters:build_adapters `
	--max-items 5 `
	--responses data/pilot-responses.jsonl `
	--report data/pilot-evaluation-report.json
```

The adapter module must be importable by Python and return one adapter per
model. Use `--generate-only` to collect responses without downloading the NLI
and embedding models. Use `--resume` with the same files, adapter metadata, and
item selection to continue an interrupted generation run; mismatched dataset
hashes or model configurations are rejected. To evaluate saved responses
without generating again, use `--evaluate-only --responses ...`. Use
`--max-items 300` and new output paths only after the benchmark is reviewed and
frozen. If no freeze manifest exists, the runner warns that results are pilot
outputs and should not be treated as final research results.

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
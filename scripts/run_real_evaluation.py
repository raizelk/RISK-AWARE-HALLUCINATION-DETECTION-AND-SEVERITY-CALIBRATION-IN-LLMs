"""Run a small provider-pluggable benchmark pilot or evaluate saved responses."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hallucination_detector.annotations import GoldAnnotations, load_adjudicated_annotations
from hallucination_detector.calibration import evaluate_severity
from hallucination_detector.domain import ModelResponse
from hallucination_detector.evaluation import ModelComparisonEvaluator
from hallucination_detector.generation import JsonlResponseStore, LLMAdapter
from hallucination_detector.retrieval import HybridEvidenceRetriever, SentenceTransformerEmbedder
from hallucination_detector.runner import load_benchmark, run_generation, write_json_atomic
from hallucination_detector.severity import SeverityClassifier
from hallucination_detector.verification import EmbeddingSimilarityScorer, HuggingFaceNLIModel


def _load_adapters(factory_spec: str) -> tuple[LLMAdapter, ...]:
    module_name, separator, function_name = factory_spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Adapter factory must use module.path:function_name syntax.")
    factory: Any = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ValueError(f"Adapter factory {factory_spec!r} is not callable.")
    adapters = tuple(factory())
    if not adapters:
        raise ValueError("Adapter factory must return at least one adapter.")
    return adapters


def _read_evidence_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            evidence_id = record.get("evidence_id") if isinstance(record, dict) else None
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError(f"Evidence line {line_number} has no valid evidence_id.")
            ids.add(evidence_id)
    return ids


def _restrict_gold(gold: GoldAnnotations, item_ids: set[str]) -> GoldAnnotations:
    return GoldAnnotations(
        labels={key: value for key, value in gold.labels.items() if key[1] in item_ids},
        evidence_ids={key: value for key, value in gold.evidence_ids.items() if key[1] in item_ids},
        severity={key: value for key, value in gold.severity.items() if key[1] in item_ids},
        annotations=tuple(item for item in gold.annotations if item.dataset_item_id in item_ids),
    )


def _evaluate(
    responses: tuple[ModelResponse, ...],
    benchmark: Any,
    *,
    top_k: int,
    nli_model_name: str,
    embedding_model_name: str,
    support_threshold: float,
    contradiction_threshold: float,
    severity_contradiction_threshold: float,
    gold_annotations: GoldAnnotations | None,
    report_path: Path,
) -> None:
    embedder = SentenceTransformerEmbedder(embedding_model_name)
    retriever = HybridEvidenceRetriever(benchmark.evidence, embedder)
    evaluator = ModelComparisonEvaluator(
        model_names=tuple(dict.fromkeys(response.model_name for response in responses)),
        dataset_item_ids=tuple(prompt.dataset_item_id for prompt in benchmark.prompts),
        retriever=retriever,
        nli_model=HuggingFaceNLIModel(nli_model_name),
        similarity_scorer=EmbeddingSimilarityScorer(embedder),
        top_k=top_k,
        support_threshold=support_threshold,
        contradiction_threshold=contradiction_threshold,
        severity_classifier=SeverityClassifier(
            high_contradiction_threshold=severity_contradiction_threshold
        ),
    )
    report = evaluator.evaluate(
        responses,
        gold_labels=gold_annotations.labels if gold_annotations else None,
        gold_evidence_ids=gold_annotations.evidence_ids if gold_annotations else None,
    )
    report.save_json(report_path)
    severity_metrics = None
    if gold_annotations is not None:
        predicted_severity = {
            (
                evaluated_response.response.model_name,
                evaluated_response.response.dataset_item_id,
                evaluated_claim.verification.claim.id,
            ): evaluated_claim.severity.severity
            for evaluated_response in report.responses
            for evaluated_claim in evaluated_response.claims
        }
        severity_metrics = {}
        for model in report.model_metrics:
            model_gold = {
                key: value
                for key, value in gold_annotations.severity.items()
                if key[0] == model.model_name
            }
            if any(value is not None for value in model_gold.values()):
                model_predictions = {
                    key: value
                    for key, value in predicted_severity.items()
                    if key[0] == model.model_name
                }
                metrics = evaluate_severity(model_gold, model_predictions)
                severity_metrics[model.model_name] = {
                    "accuracy": metrics.accuracy,
                    "macro_f1": metrics.macro_f1,
                    "quadratic_weighted_kappa": metrics.quadratic_weighted_kappa,
                    "sample_size": metrics.sample_size,
                    "confusion_matrix": metrics.confusion_matrix,
                }
    write_json_atomic(
        report_path.with_suffix(report_path.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "prompt_sha256": benchmark.prompt_sha256,
            "evidence_sha256": benchmark.evidence_sha256,
            "dataset_item_ids": list(report.dataset_item_ids),
            "models": [metrics.model_name for metrics in report.model_metrics],
            "top_k": top_k,
            "support_threshold": support_threshold,
            "contradiction_threshold": contradiction_threshold,
            "severity_contradiction_threshold": severity_contradiction_threshold,
            "nli_model": nli_model_name,
            "embedding_model": embedding_model_name,
            "benchmark_frozen": benchmark.is_frozen,
            "severity_metrics_vs_gold": severity_metrics,
        },
    )
    print(f"Saved evaluation report: {report_path}")
    for metrics in report.model_metrics:
        print(
            f"{metrics.model_name}: claims={metrics.claim_count}, "
            f"contradicted_rate={metrics.contradicted_rate}, "
            f"unknown_rate={metrics.unknown_rate}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/healthcare/prompts.jsonl")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "data/healthcare/trusted_evidence.jsonl"
    )
    parser.add_argument("--responses", type=Path, default=ROOT / "data/pilot-responses.jsonl")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "data/pilot-evaluation-report.json"
    )
    parser.add_argument(
        "--adapter-factory",
        help="Python factory in module.path:function_name form; required for generation",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="number of prompts for generation (default: 5; use 300 for the full Phase 1 set)",
    )
    parser.add_argument("--resume", action="store_true", help="resume a matching generation run")
    parser.add_argument(
        "--generate-only", action="store_true", help="save responses without running verification"
    )
    parser.add_argument(
        "--evaluate-only", action="store_true", help="evaluate the existing response JSONL"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--support-threshold", type=float, default=0.55)
    parser.add_argument("--contradiction-threshold", type=float, default=0.55)
    parser.add_argument("--severity-contradiction-threshold", type=float, default=0.8)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--dataset-split", choices=("train", "dev", "test"))
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args(argv)

    if args.generate_only and args.evaluate_only:
        parser.error("--generate-only and --evaluate-only cannot be combined.")
    if args.evaluate_only and args.adapter_factory:
        parser.error("--adapter-factory is not used with --evaluate-only.")
    if args.annotations is not None and not args.evaluate_only:
        parser.error("--annotations is only supported with --evaluate-only.")
    if args.resume and args.evaluate_only:
        parser.error("--resume applies only to generation.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if (args.split_manifest is None) != (args.dataset_split is None):
        parser.error("--split-manifest and --dataset-split must be provided together.")
    if not args.evaluate_only and not args.adapter_factory:
        parser.error("Provide --adapter-factory to generate responses, or use --evaluate-only.")

    try:
        benchmark = load_benchmark(args.prompts, args.evidence)
        if not benchmark.is_frozen:
            print("WARNING: benchmark has no freeze manifest; treat this as a pilot, not a final run.")
        all_item_ids = {prompt.dataset_item_id for prompt in benchmark.prompts}
        if args.split_manifest is not None:
            split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
            if not isinstance(split_manifest, dict) or split_manifest.get("schema_version") != 1:
                raise ValueError("Split manifest must be a schema_version 1 JSON object.")
            if split_manifest.get("prompt_sha256") != benchmark.prompt_sha256:
                raise ValueError("Split manifest prompt hash does not match the benchmark.")
            if split_manifest.get("evidence_sha256") != benchmark.evidence_sha256:
                raise ValueError("Split manifest evidence hash does not match the benchmark.")
            item_splits = split_manifest.get("item_splits")
            if not isinstance(item_splits, dict):
                raise ValueError("Split manifest is missing item_splits.")
            expected_items = {prompt.dataset_item_id for prompt in benchmark.prompts}
            if set(item_splits) != expected_items:
                raise ValueError("Split manifest does not cover the current benchmark items.")
            if any(value not in {"train", "dev", "test"} for value in item_splits.values()):
                raise ValueError("Split manifest contains an invalid split name.")
            benchmark = replace(
                benchmark,
                prompts=tuple(
                    prompt for prompt in benchmark.prompts
                    if item_splits[prompt.dataset_item_id] == args.dataset_split
                ),
            )
            if not benchmark.prompts:
                raise ValueError(f"The {args.dataset_split} split has no prompts.")

        if not args.evaluate_only:
            if args.max_items < 1:
                parser.error("--max-items must be at least 1.")
            benchmark = replace(benchmark, prompts=benchmark.prompts[:args.max_items])

        if args.evaluate_only:
            responses = JsonlResponseStore().load(args.responses)
            gold_annotations = None
            if args.annotations is not None:
                full_gold = load_adjudicated_annotations(
                    args.annotations,
                    responses,
                    _read_evidence_ids(args.evidence),
                    require_complete=True,
                )
                gold_annotations = full_gold
            response_item_ids = {response.dataset_item_id for response in responses}
            benchmark_item_ids = {prompt.dataset_item_id for prompt in benchmark.prompts}
            unexpected = response_item_ids - all_item_ids
            if unexpected:
                raise ValueError(f"Responses contain unknown dataset items: {sorted(unexpected)!r}.")
            selected_item_ids = response_item_ids & benchmark_item_ids
            benchmark = replace(
                benchmark,
                prompts=tuple(
                    prompt for prompt in benchmark.prompts
                    if prompt.dataset_item_id in selected_item_ids
                ),
            )
            responses = tuple(
                response for response in responses
                if response.dataset_item_id in selected_item_ids
            )
            if gold_annotations is not None:
                gold_annotations = _restrict_gold(gold_annotations, selected_item_ids)
            if not benchmark.prompts:
                raise ValueError("The response file has no dataset items from this benchmark.")
        else:
            adapters = _load_adapters(args.adapter_factory)
            responses = run_generation(
                benchmark.prompts,
                adapters,
                args.responses,
                prompt_sha256=benchmark.prompt_sha256,
                evidence_sha256=benchmark.evidence_sha256,
                resume=args.resume,
            )
            print(f"Saved {len(responses)} responses: {args.responses}")

        if not args.generate_only:
            _evaluate(
                responses,
                benchmark,
                top_k=args.top_k,
                nli_model_name=args.nli_model,
                embedding_model_name=args.embedding_model,
                support_threshold=args.support_threshold,
                contradiction_threshold=args.contradiction_threshold,
                severity_contradiction_threshold=args.severity_contradiction_threshold,
                gold_annotations=gold_annotations if args.evaluate_only else None,
                report_path=args.report,
            )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Run a small provider-pluggable benchmark pilot or evaluate saved responses."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hallucination_detector.domain import ModelResponse
from hallucination_detector.evaluation import ModelComparisonEvaluator
from hallucination_detector.generation import JsonlResponseStore, LLMAdapter
from hallucination_detector.retrieval import HybridEvidenceRetriever, SentenceTransformerEmbedder
from hallucination_detector.runner import load_benchmark, run_generation, write_json_atomic
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


def _evaluate(
    responses: tuple[ModelResponse, ...],
    benchmark: Any,
    *,
    top_k: int,
    nli_model_name: str,
    embedding_model_name: str,
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
    )
    report = evaluator.evaluate(responses)
    report.save_json(report_path)
    write_json_atomic(
        report_path.with_suffix(report_path.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "prompt_sha256": benchmark.prompt_sha256,
            "evidence_sha256": benchmark.evidence_sha256,
            "dataset_item_ids": list(report.dataset_item_ids),
            "models": [metrics.model_name for metrics in report.model_metrics],
            "top_k": top_k,
            "nli_model": nli_model_name,
            "embedding_model": embedding_model_name,
            "benchmark_frozen": benchmark.is_frozen,
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
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args(argv)

    if args.generate_only and args.evaluate_only:
        parser.error("--generate-only and --evaluate-only cannot be combined.")
    if args.evaluate_only and args.adapter_factory:
        parser.error("--adapter-factory is not used with --evaluate-only.")
    if args.resume and args.evaluate_only:
        parser.error("--resume applies only to generation.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if not args.evaluate_only and not args.adapter_factory:
        parser.error("Provide --adapter-factory to generate responses, or use --evaluate-only.")

    max_items = None if args.evaluate_only else args.max_items
    try:
        benchmark = load_benchmark(args.prompts, args.evidence, max_items=max_items)
        if not benchmark.is_frozen:
            print("WARNING: benchmark has no freeze manifest; treat this as a pilot, not a final run.")

        if args.evaluate_only:
            responses = JsonlResponseStore().load(args.responses)
            response_item_ids = {response.dataset_item_id for response in responses}
            benchmark = type(benchmark)(
                prompts=tuple(
                    prompt for prompt in benchmark.prompts
                    if prompt.dataset_item_id in response_item_ids
                ),
                evidence=benchmark.evidence,
                prompt_sha256=benchmark.prompt_sha256,
                evidence_sha256=benchmark.evidence_sha256,
                is_frozen=benchmark.is_frozen,
            )
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
                report_path=args.report,
            )
    except (ImportError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
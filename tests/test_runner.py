import json

import pytest

from hallucination_detector.generation import GeneratedText
from hallucination_detector.runner import load_benchmark, run_generation


class CountingAdapter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = 0
        self.metadata = {"provider": "test", "model_version": "test-v1"}

    def generate(self, record):
        self.calls += 1
        return GeneratedText(f"{self.model_name} response to {record.dataset_item_id}")


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _benchmark_files(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    evidence = tmp_path / "trusted_evidence.jsonl"
    _write_jsonl(
        prompts,
        [
            {
                "dataset_item_id": f"item-{index}",
                "topic": "topic-a",
                "prompt": f"Question {index}?",
                "reference_evidence_ids": ["evidence-a"],
            }
            for index in (1, 2)
        ],
    )
    _write_jsonl(
        evidence,
        [
            {
                "evidence_id": "evidence-a",
                "topic": "topic-a",
                "text": "Reference passage.",
                "source": "Test source",
            }
        ],
    )
    return prompts, evidence


def test_load_benchmark_checks_links_and_applies_pilot_limit(tmp_path):
    prompts_path, evidence_path = _benchmark_files(tmp_path)

    benchmark = load_benchmark(prompts_path, evidence_path, max_items=1)

    assert [prompt.dataset_item_id for prompt in benchmark.prompts] == ["item-1"]
    assert benchmark.evidence[0].id == "evidence-a"
    assert len(benchmark.prompt_sha256) == 64
    assert len(benchmark.evidence_sha256) == 64
    assert benchmark.is_frozen is False


def test_generation_checkpoints_and_matching_resume_skips_completed_calls(tmp_path):
    prompts_path, evidence_path = _benchmark_files(tmp_path)
    benchmark = load_benchmark(prompts_path, evidence_path)
    adapters = (CountingAdapter("model-a"), CountingAdapter("model-b"))
    response_path = tmp_path / "responses.jsonl"

    first_run = run_generation(
        benchmark.prompts,
        adapters,
        response_path,
        prompt_sha256=benchmark.prompt_sha256,
        evidence_sha256=benchmark.evidence_sha256,
    )
    resumed_run = run_generation(
        benchmark.prompts,
        adapters,
        response_path,
        prompt_sha256=benchmark.prompt_sha256,
        evidence_sha256=benchmark.evidence_sha256,
        resume=True,
    )

    assert len(first_run) == 4
    assert resumed_run == first_run
    assert [adapter.calls for adapter in adapters] == [2, 2]
    manifest = json.loads(
        response_path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["completed_response_count"] == 4
    assert manifest["adapters"][0]["metadata"]["model_version"] == "test-v1"


def test_resume_rejects_changed_dataset_fingerprint(tmp_path):
    prompts_path, evidence_path = _benchmark_files(tmp_path)
    benchmark = load_benchmark(prompts_path, evidence_path)
    response_path = tmp_path / "responses.jsonl"
    adapters = (CountingAdapter("model-a"),)
    run_generation(
        benchmark.prompts,
        adapters,
        response_path,
        prompt_sha256=benchmark.prompt_sha256,
        evidence_sha256=benchmark.evidence_sha256,
    )

    with pytest.raises(ValueError, match="prompt_sha256"):
        run_generation(
            benchmark.prompts,
            adapters,
            response_path,
            prompt_sha256="changed",
            evidence_sha256=benchmark.evidence_sha256,
            resume=True,
        )


def test_generation_failure_is_checkpointed_without_error_message_and_can_resume(tmp_path):
    class FailOnceAdapter(CountingAdapter):
        def __init__(self):
            super().__init__("model-a")
            self.fail_on_second_call = True

        def generate(self, record):
            self.calls += 1
            if self.calls == 2 and self.fail_on_second_call:
                raise RuntimeError("provider error containing private details")
            return GeneratedText(f"response to {record.dataset_item_id}")

    prompts_path, evidence_path = _benchmark_files(tmp_path)
    benchmark = load_benchmark(prompts_path, evidence_path)
    response_path = tmp_path / "responses.jsonl"
    adapter = FailOnceAdapter()
    run_args = {
        "prompts": benchmark.prompts,
        "adapters": (adapter,),
        "response_path": response_path,
        "prompt_sha256": benchmark.prompt_sha256,
        "evidence_sha256": benchmark.evidence_sha256,
    }

    with pytest.raises(RuntimeError, match="private details"):
        run_generation(**run_args)

    manifest_path = response_path.with_suffix(".jsonl.manifest.json")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["completed_response_count"] == 1
    assert manifest["last_error"]["dataset_item_id"] == "item-2"
    assert "private details" not in manifest_text
    assert manifest["completed_response_metadata"][0]["generated_at_utc"]

    adapter.fail_on_second_call = False
    resumed = run_generation(**run_args, resume=True)

    assert len(resumed) == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["complete"] is True


def test_load_benchmark_rejects_topic_mismatched_evidence(tmp_path):
    prompts_path, evidence_path = _benchmark_files(tmp_path)
    prompts = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    prompts[0]["topic"] = "different-topic"
    _write_jsonl(prompts_path, prompts)

    with pytest.raises(ValueError, match="topic does not match"):
        load_benchmark(prompts_path, evidence_path)
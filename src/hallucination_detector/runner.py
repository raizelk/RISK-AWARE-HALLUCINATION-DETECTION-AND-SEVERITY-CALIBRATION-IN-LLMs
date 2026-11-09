"""Dataset loading and resumable model-response generation for benchmark runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import Evidence, ModelResponse
from .generation import JsonlResponseStore, LLMAdapter, PromptRecord, ResponseGenerator


@dataclass(frozen=True)
class Benchmark:
    prompts: tuple[PromptRecord, ...]
    evidence: tuple[Evidence, ...]
    prompt_sha256: str
    evidence_sha256: str
    is_frozen: bool


def load_benchmark(
    prompts_path: str | Path,
    evidence_path: str | Path,
    *,
    max_items: int | None = None,
) -> Benchmark:
    """Load JSONL inputs and verify IDs, references, and any freeze manifest."""
    prompt_file = Path(prompts_path)
    evidence_file = Path(evidence_path)
    prompt_records = _load_jsonl(prompt_file)
    evidence_records = _load_jsonl(evidence_file)

    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be at least 1 when provided.")
    prompts: list[PromptRecord] = []
    prompt_ids: set[str] = set()
    for line_number, record in enumerate(prompt_records, start=1):
        item_id = _required_text(record, "dataset_item_id", prompt_file, line_number)
        prompt = _required_text(record, "prompt", prompt_file, line_number)
        if item_id in prompt_ids:
            raise ValueError(f"{prompt_file.name}:{line_number}: duplicate item ID {item_id!r}")
        prompt_ids.add(item_id)
        prompts.append(PromptRecord(item_id, prompt))

    evidence: list[Evidence] = []
    evidence_ids: set[str] = set()
    evidence_topic: dict[str, str] = {}
    for line_number, record in enumerate(evidence_records, start=1):
        evidence_id = _required_text(record, "evidence_id", evidence_file, line_number)
        text = _required_text(record, "text", evidence_file, line_number)
        source = _required_text(record, "source", evidence_file, line_number)
        topic = _required_text(record, "topic", evidence_file, line_number)
        if evidence_id in evidence_ids:
            raise ValueError(
                f"{evidence_file.name}:{line_number}: duplicate evidence ID {evidence_id!r}"
            )
        evidence_ids.add(evidence_id)
        evidence_topic[evidence_id] = topic
        evidence.append(Evidence(evidence_id, text, source))

    for line_number, record in enumerate(prompt_records, start=1):
        references = record.get("reference_evidence_ids")
        if not isinstance(references, list) or not references:
            raise ValueError(
                f"{prompt_file.name}:{line_number}: reference_evidence_ids must be a non-empty array"
            )
        topic = record.get("topic")
        for evidence_id in references:
            if evidence_id not in evidence_ids:
                raise ValueError(
                    f"{prompt_file.name}:{line_number}: unknown evidence ID {evidence_id!r}"
                )
            if evidence_topic[evidence_id] != topic:
                raise ValueError(
                    f"{prompt_file.name}:{line_number}: linked evidence topic does not match prompt topic"
                )

    _validate_freeze_manifest(prompt_file, evidence_file)
    if max_items is not None:
        prompts = prompts[:max_items]

    return Benchmark(
        prompts=tuple(prompts),
        evidence=tuple(evidence),
        prompt_sha256=_sha256(prompt_file),
        evidence_sha256=_sha256(evidence_file),
        is_frozen=(prompt_file.parent / "benchmark_manifest.json").exists(),
    )


def run_generation(
    prompts: Sequence[PromptRecord],
    adapters: tuple[LLMAdapter, ...],
    response_path: str | Path,
    *,
    prompt_sha256: str,
    evidence_sha256: str,
    resume: bool = False,
) -> tuple[ModelResponse, ...]:
    """Generate and checkpoint each response, allowing a matching run to resume."""
    if not prompts:
        raise ValueError("At least one prompt is required.")
    ResponseGenerator(adapters)
    target = Path(response_path)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    adapter_records = [_adapter_manifest(adapter) for adapter in adapters]
    expected_manifest = {
        "schema_version": 1,
        "prompt_sha256": prompt_sha256,
        "evidence_sha256": evidence_sha256,
        "dataset_item_ids": [prompt.dataset_item_id for prompt in prompts],
        "adapters": adapter_records,
    }

    existing: list[ModelResponse] = []
    saved_manifest: dict[str, Any] | None = None
    if target.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Run files already exist for {target}; pass resume=True to continue the same run."
            )
        if not manifest_path.exists():
            raise ValueError("Cannot resume responses without their run manifest.")
        saved_manifest = _read_json(manifest_path)
        for key, value in expected_manifest.items():
            if saved_manifest.get(key) != value:
                raise ValueError(f"Cannot resume: run manifest does not match {key}.")
        if target.exists():
            existing = list(JsonlResponseStore().load(target))

    model_names = {adapter.model_name for adapter in adapters}
    item_ids = {prompt.dataset_item_id for prompt in prompts}
    response_keys: set[tuple[str, str]] = set()
    for response in existing:
        key = (response.model_name, response.dataset_item_id)
        if key in response_keys:
            raise ValueError(f"Duplicate response already exists for {key!r}.")
        if response.model_name not in model_names or response.dataset_item_id not in item_ids:
            raise ValueError(f"Existing response is outside this run: {key!r}.")
        response_keys.add(key)

    completed_metadata = (
        saved_manifest.get("completed_response_metadata", [])
        if saved_manifest is not None
        else []
    )
    if not isinstance(completed_metadata, list):
        raise ValueError("Run manifest has invalid completed-response metadata.")
    metadata_keys = {
        (item.get("model_name"), item.get("dataset_item_id"))
        for item in completed_metadata
        if isinstance(item, dict)
    }
    if not metadata_keys <= response_keys:
        raise ValueError("Run manifest contains metadata for a response that was not saved.")
    for response in existing:
        key = (response.model_name, response.dataset_item_id)
        if key not in metadata_keys:
            completed_metadata.append(
                {
                    "model_name": response.model_name,
                    "dataset_item_id": response.dataset_item_id,
                    "generated_at_utc": None,
                    "recovered_after_interruption": True,
                }
            )
            metadata_keys.add(key)

    manifest: dict[str, Any] = {
        **expected_manifest,
        "expected_response_count": len(prompts) * len(adapters),
        "completed_response_count": len(existing),
        "completed_response_metadata": completed_metadata,
        "complete": False,
        "created_at_utc": (
            saved_manifest.get("created_at_utc")
            if saved_manifest is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)

    store = JsonlResponseStore()
    generators = tuple(ResponseGenerator((adapter,)) for adapter in adapters)
    for prompt in prompts:
        for adapter, one_model_generator in zip(adapters, generators):
            key = (adapter.model_name, prompt.dataset_item_id)
            if key in response_keys:
                continue
            try:
                response = one_model_generator.generate((prompt,))[0]
            except Exception as error:
                manifest["last_error"] = {
                    "model_name": adapter.model_name,
                    "dataset_item_id": prompt.dataset_item_id,
                    "error_type": type(error).__name__,
                    "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json_atomic(manifest_path, manifest)
                raise
            existing.append(response)
            response_keys.add(key)
            store.save(target, tuple(existing))
            manifest["completed_response_metadata"].append(
                {
                    "model_name": response.model_name,
                    "dataset_item_id": response.dataset_item_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            manifest.pop("last_error", None)
            manifest["completed_response_count"] = len(existing)
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(manifest_path, manifest)

    manifest["complete"] = len(response_keys) == manifest["expected_response_count"]
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)
    return tuple(existing)


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write a JSON sidecar such as evaluation configuration."""
    _write_json_atomic(Path(path), dict(payload))


def _adapter_manifest(adapter: LLMAdapter) -> dict[str, object]:
    metadata = getattr(adapter, "metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Adapter {adapter.model_name!r} metadata must be a mapping.")
    record = {
        "model_name": adapter.model_name,
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "metadata": dict(metadata),
    }
    try:
        json.dumps(record)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Adapter {adapter.model_name!r} metadata must be JSON serializable.") from error
    return record


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path.name} line {line_number}.") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} line {line_number} must be a JSON object.")
            records.append(value)
    return records


def _required_text(
    record: Mapping[str, object], field: str, path: Path, line_number: int
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.name}:{line_number}: {field} must be non-empty text")
    return value


def _validate_freeze_manifest(prompt_file: Path, evidence_file: Path) -> None:
    manifest_path = prompt_file.parent / "benchmark_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Benchmark manifest is missing its files hash mapping.")
    for source in (prompt_file, evidence_file):
        expected = files.get(source.name)
        if expected != _sha256(source):
            raise ValueError(f"Benchmark file {source.name!r} differs from its freeze manifest.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path} must contain an object.")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
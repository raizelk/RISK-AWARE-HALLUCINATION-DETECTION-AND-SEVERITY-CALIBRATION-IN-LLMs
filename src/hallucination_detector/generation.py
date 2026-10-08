"""Model response generation and persistence for Sprint 2.

The core deliberately does not depend on a provider SDK. Provider-specific
adapters can implement :class:`LLMAdapter`, while tests and local experiments
can use :class:`StaticLLMAdapter`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

from .domain import ModelResponse


@dataclass(frozen=True)
class PromptRecord:
    """A dataset item ready to be sent to a language model."""

    dataset_item_id: str
    prompt: str


@dataclass(frozen=True)
class GeneratedText:
    """Text and optional self-reported confidence returned by an adapter."""

    text: str
    confidence: Optional[float] = None


class LLMAdapter(Protocol):
    """Provider-neutral interface required by the batch generator."""

    @property
    def model_name(self) -> str: ...

    def generate(self, record: PromptRecord) -> GeneratedText: ...


@dataclass(frozen=True)
class StaticLLMAdapter:
    """Deterministic adapter useful for tests and offline benchmark fixtures."""

    model_name: str
    responses: dict[str, GeneratedText]

    def generate(self, record: PromptRecord) -> GeneratedText:
        try:
            return self.responses[record.dataset_item_id]
        except KeyError as error:
            raise KeyError(
                f"No response configured for model {self.model_name!r} "
                f"and dataset item {record.dataset_item_id!r}"
            ) from error


class ResponseGenerator:
    """Generate a complete, ordered response matrix for configured models."""

    def __init__(self, adapters: tuple[LLMAdapter, ...]) -> None:
        names = [adapter.model_name for adapter in adapters]
        if not names or any(not name.strip() for name in names):
            raise ValueError("At least one adapter with a non-empty model name is required")
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique")
        self._adapters = adapters

    def generate(self, records: tuple[PromptRecord, ...]) -> tuple[ModelResponse, ...]:
        responses: list[ModelResponse] = []
        for record in records:
            if not record.dataset_item_id.strip():
                raise ValueError("Dataset item IDs must be non-empty")
            if not record.prompt.strip():
                raise ValueError(f"Prompt for {record.dataset_item_id!r} must be non-empty")
            for adapter in self._adapters:
                generated = adapter.generate(record)
                if not generated.text.strip():
                    raise ValueError(
                        f"Model {adapter.model_name!r} returned an empty response "
                        f"for {record.dataset_item_id!r}"
                    )
                if generated.confidence is not None and not 0 <= generated.confidence <= 1:
                    raise ValueError("Confidence must be between 0 and 1")
                responses.append(
                    ModelResponse(
                        model_name=adapter.model_name,
                        dataset_item_id=record.dataset_item_id,
                        text=generated.text,
                        confidence=generated.confidence,
                    )
                )
        return tuple(responses)


class JsonlResponseStore:
    """Persist model responses as one JSON object per line."""

    def save(self, path: Union[str, Path], responses: tuple[ModelResponse, ...]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for response in responses:
                    stream.write(json.dumps(asdict(response), ensure_ascii=True) + "\n")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path: Union[str, Path]) -> tuple[ModelResponse, ...]:
        source = Path(path)
        responses: list[ModelResponse] = []
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    responses.append(ModelResponse(**value))
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(f"Invalid response record on line {line_number}") from error
        return tuple(responses)

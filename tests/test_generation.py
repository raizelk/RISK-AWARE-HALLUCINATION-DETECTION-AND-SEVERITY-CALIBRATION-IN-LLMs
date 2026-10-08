import pytest

from hallucination_detector.domain import ModelResponse
from hallucination_detector.generation import (
    GeneratedText,
    JsonlResponseStore,
    PromptRecord,
    ResponseGenerator,
    StaticLLMAdapter,
)


def _generator() -> ResponseGenerator:
    return ResponseGenerator(
        (
            StaticLLMAdapter("model-a", {"item-1": GeneratedText("Canberra.", 0.9)}),
            StaticLLMAdapter("model-b", {"item-1": GeneratedText("Sydney.", 0.8)}),
            StaticLLMAdapter("model-c", {"item-1": GeneratedText("Canberra.", 0.95)}),
        )
    )


def test_generator_creates_one_response_per_model_in_stable_order():
    responses = _generator().generate((PromptRecord("item-1", "Capital?"),))

    assert [(item.model_name, item.text) for item in responses] == [
        ("model-a", "Canberra."),
        ("model-b", "Sydney."),
        ("model-c", "Canberra."),
    ]


def test_jsonl_store_round_trips_model_responses(tmp_path):
    expected = _generator().generate((PromptRecord("item-1", "Capital?"),))
    path = tmp_path / "responses.jsonl"

    store = JsonlResponseStore()
    store.save(path, expected)

    assert store.load(path) == expected


def test_generator_rejects_invalid_confidence():
    generator = ResponseGenerator(
        (StaticLLMAdapter("model-a", {"item-1": GeneratedText("answer", 1.1)}),)
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        generator.generate((PromptRecord("item-1", "question"),))


def test_store_rejects_malformed_record(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text('{"model_name": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        JsonlResponseStore().load(path)


def test_model_response_contract_remains_json_friendly():
    response = ModelResponse("model-a", "item-1", "answer", 0.5)
    assert response.dataset_item_id == "item-1"

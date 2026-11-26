import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from hallucination_detector.generation import PromptRecord
from hallucination_detector.provider_adapters import (
    GenerationSettings,
    GeminiAdapter,
    OpenAICompatibleAdapter,
    ProviderRequestError,
    create_default_adapters,
    create_ollama_adapter,
)


class FakeHttpResponse:
    def __init__(self, value):
        self._body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._body


def test_gemini_adapter_uses_identical_prompt_and_shared_generation_settings():
    settings = GenerationSettings(temperature=0.1, top_p=0.9, max_output_tokens=256)
    adapter = GeminiAdapter("local-key", settings)
    prompt = PromptRecord("prompt-1", "Exact source question?", "topic-1")

    with patch(
        "hallucination_detector.provider_adapters.urllib.request.urlopen",
        return_value=FakeHttpResponse(
            {"candidates": [{"content": {"parts": [{"text": "Model answer."}]}}]}
        ),
    ) as urlopen:
        generated = adapter.generate(prompt)

    request = urlopen.call_args.args[0]
    request_body = json.loads(request.data)
    assert generated.text == "Model answer."
    assert request_body["contents"] == [
        {"role": "user", "parts": [{"text": "Exact source question?"}]}
    ]
    assert request_body["generationConfig"] == {
        "temperature": 0.1,
        "topP": 0.9,
        "maxOutputTokens": 256,
    }
    assert request.get_header("X-goog-api-key") == "local-key"
    assert "local-key" not in json.dumps(adapter.metadata)


def test_openai_compatible_adapters_use_shared_parameters_and_prompt():
    settings = GenerationSettings(temperature=0.2, top_p=1.0, max_output_tokens=1024)
    adapter = OpenAICompatibleAdapter(
        model_name="openai/gpt-oss-120b",
        provider="groq",
        api_key="local-key",
        base_url="https://api.groq.com/openai/v1",
        generation_settings=settings,
    )

    with patch(
        "hallucination_detector.provider_adapters.urllib.request.urlopen",
        return_value=FakeHttpResponse(
            {"choices": [{"message": {"content": "Model answer."}}]}
        ),
    ) as urlopen:
        generated = adapter.generate(PromptRecord("prompt-1", "Exact source question?"))

    request = urlopen.call_args.args[0]
    request_body = json.loads(request.data)
    assert generated.text == "Model answer."
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request_body == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Exact source question?"}],
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1024,
    }


def test_provider_failure_redacts_credential_from_error():
    adapter = GeminiAdapter("private-key")

    with patch(
        "hallucination_detector.provider_adapters.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://provider.invalid",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"error":"private-key is invalid"}'),
        ),
    ):
        with pytest.raises(ProviderRequestError) as error:
            adapter.generate(PromptRecord("prompt-1", "Question?"))

    assert "private-key" not in str(error.value)


def test_default_adapters_require_local_credentials_and_use_three_pinned_models(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
        create_default_adapters()

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    adapters = create_default_adapters(GenerationSettings())

    assert [adapter.model_name for adapter in adapters] == [
        "gemini-2.5-flash",
        "openai/gpt-oss-120b",
        "deepseek/deepseek-r1:free",
    ]
    assert all(
        adapter.metadata["generation_parameters"] == adapters[0].metadata["generation_parameters"]
        for adapter in adapters
    )
    assert all(
        key not in json.dumps(adapter.metadata)
        for adapter, key in zip(adapters, ("gemini-secret", "groq-secret", "openrouter-secret"))
    )


def test_ollama_adapter_uses_local_defaults_and_shared_generation_settings(monkeypatch):
    settings = GenerationSettings(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=128,
        timeout_seconds=300,
    )
    (adapter,) = create_ollama_adapter(settings)

    with patch(
        "hallucination_detector.provider_adapters.urllib.request.urlopen",
        return_value=FakeHttpResponse(
            {"choices": [{"message": {"content": "Local answer."}}]}
        ),
    ) as urlopen:
        generated = adapter.generate(PromptRecord("prompt-1", "Question?"))

    request_body = json.loads(urlopen.call_args.args[0].data)
    assert generated.text == "Local answer."
    assert adapter.model_name == "qwen2.5:3b"
    assert adapter.metadata["provider"] == "ollama"
    assert adapter.metadata["generation_parameters"] == settings.as_dict()
    assert request_body == {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": "Question?"}],
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 128,
    }

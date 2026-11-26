import os
from unittest.mock import patch

import pytest

import live_adapters
from hallucination_detector.generation import GeneratedText, PromptRecord


def test_probe_adapters_returns_statuses_without_returning_responses():
    class FakeAdapter:
        metadata = {"provider": "fake", "model_version": "test"}

        def generate(self, record):
            return GeneratedText("ready")

    statuses = live_adapters.probe_adapters((FakeAdapter(),))

    assert statuses == {"fake": "PASS"}


def test_dotenv_loader_sets_missing_values_and_preserves_shell(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TEST_DOTENV_VALUE=file-value\nTEST_DOTENV_EXISTING=shell-should-win\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DOTENV_EXISTING", "shell-value")
    monkeypatch.delenv("TEST_DOTENV_VALUE", raising=False)

    live_adapters._load_dotenv(env_file)

    assert os.getenv("TEST_DOTENV_VALUE") == "file-value"
    assert os.getenv("TEST_DOTENV_EXISTING") == "shell-value"


def test_build_adapters_requires_credentials(monkeypatch):
    for name in (
        "OPENROUTER_API_KEY", "HY3_API_KEY", "GROQ_API_KEY", "GROK_API_KEY",
        "ORCAROUTER_API_KEY", "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="HY3_API_KEY"):
        live_adapters.build_adapters()


def test_build_adapters_reads_env_and_never_includes_keys_in_metadata(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "hy3-test")
    monkeypatch.setenv("GROQ_MODEL", "groq-test")
    monkeypatch.setenv("ORCAROUTER_MODEL", "deepseek-test")

    adapters = live_adapters.build_adapters()

    assert [adapter.model_name for adapter in adapters] == ["hy3-test", "groq-test", "deepseek-test"]
    assert all("secret" not in str(adapter.metadata) for adapter in adapters)


def test_build_ollama_adapter_uses_local_defaults_and_labels_provenance(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    (adapter,) = live_adapters.build_ollama_adapter()

    assert adapter.model_name == "qwen2.5:3b"
    assert adapter.base_url == "http://127.0.0.1:11434/v1"
    assert adapter.provider == "ollama"
    assert adapter.metadata == {
        "provider": "ollama",
        "model_version": "qwen2.5:3b",
        "research_use": False,
    }


def test_hy3_adapter_parses_chat_content():
    adapter = live_adapters.OpenAICompatibleAdapter(
        model_name="hy3-test",
        api_key="secret",
        base_url="https://example.test/v1",
        provider="hy3",
        temperature=0.0,
        timeout=30.0,
        metadata={"provider": "hy3"},
    )
    payload = {"choices": [{"message": {"content": "HY3 answer"}}]}

    with patch("live_adapters._post_json", return_value=payload):
        result = adapter.generate(PromptRecord("item-1", "Question?"))

    assert result.text == "HY3 answer"


def test_openai_compatible_adapter_parses_chat_content():
    adapter = live_adapters.OpenAICompatibleAdapter(
        model_name="groq-test",
        api_key="secret",
        base_url="https://example.test/v1",
        provider="groq",
        temperature=0.0,
        timeout=30.0,
        metadata={"provider": "groq"},
    )
    payload = {"choices": [{"message": {"content": "Groq answer"}}]}

    with patch("live_adapters._post_json", return_value=payload):
        result = adapter.generate(PromptRecord("item-1", "Question?"))

    assert result.text == "Groq answer"

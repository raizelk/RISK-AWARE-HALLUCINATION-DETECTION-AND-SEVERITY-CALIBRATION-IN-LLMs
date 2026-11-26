"""HTTP adapters for the three configured response-generation providers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .generation import GeneratedText, PromptRecord


class ProviderRequestError(RuntimeError):
    """A provider request failed or returned an unusable response."""


@dataclass(frozen=True)
class GenerationSettings:
    """Generation options shared by every model in a comparison run."""

    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 1024
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1.")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1.")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
        }


class _ProviderAdapter:
    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        api_key: str,
        generation_settings: GenerationSettings,
    ) -> None:
        self.model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self.generation_settings = generation_settings
        self.metadata: dict[str, object] = {
            "provider": provider,
            "model_version": model_name,
            "generation_parameters": generation_settings.as_dict(),
        }

    def _post_json(self, url: str, payload: Mapping[str, object], headers: Mapping[str, str]) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.generation_settings.timeout_seconds
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            detail = detail.replace(self._api_key, "[REDACTED]")[:500]
            raise ProviderRequestError(
                f"{self._provider} returned HTTP {error.code}: {detail or error.reason}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ProviderRequestError(
                f"{self._provider} request failed ({type(error).__name__})."
            ) from None
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderRequestError(f"{self._provider} returned invalid JSON.") from None
        if not isinstance(value, dict):
            raise ProviderRequestError(f"{self._provider} returned a non-object response.")
        return value


class GeminiAdapter(_ProviderAdapter):
    """Google Gemini generateContent adapter using a local API key."""

    def __init__(
        self,
        api_key: str,
        generation_settings: GenerationSettings = GenerationSettings(),
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider="google-gemini",
            api_key=api_key,
            generation_settings=generation_settings,
        )

    def generate(self, record: PromptRecord) -> GeneratedText:
        settings = self.generation_settings
        payload = {
            "contents": [{"role": "user", "parts": [{"text": record.prompt}]}],
            "generationConfig": {
                "temperature": settings.temperature,
                "topP": settings.top_p,
                "maxOutputTokens": settings.max_output_tokens,
            },
        }
        result = self._post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            payload,
            {"x-goog-api-key": self._api_key},
        )
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderRequestError("Google Gemini returned no response candidates.")
        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text.strip():
            raise ProviderRequestError("Google Gemini returned an empty response.")
        return GeneratedText(text)


class OpenAICompatibleAdapter(_ProviderAdapter):
    """Adapter for OpenAI-compatible Groq and OpenRouter chat completions."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        api_key: str,
        base_url: str,
        generation_settings: GenerationSettings,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            generation_settings=generation_settings,
        )
        self._base_url = base_url.rstrip("/")
        self._extra_headers = dict(extra_headers or {})

    def generate(self, record: PromptRecord) -> GeneratedText:
        settings = self.generation_settings
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": record.prompt}],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_output_tokens,
        }
        result = self._post_json(
            f"{self._base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self._api_key}", **self._extra_headers},
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderRequestError(f"{self._provider} returned no completion choices.")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str) or not content.strip():
            raise ProviderRequestError(f"{self._provider} returned an empty response.")
        return GeneratedText(content)


def create_default_adapters(
    generation_settings: GenerationSettings = GenerationSettings(),
) -> tuple[_ProviderAdapter, ...]:
    """Create the three requested adapters without accepting credentials as arguments."""
    gemini_key = _required_environment_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    groq_key = _required_environment_key("GROQ_API_KEY")
    openrouter_key = _required_environment_key("OPENROUTER_API_KEY")
    return (
        GeminiAdapter(gemini_key, generation_settings),
        OpenAICompatibleAdapter(
            model_name="openai/gpt-oss-120b",
            provider="groq",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            generation_settings=generation_settings,
        ),
        OpenAICompatibleAdapter(
            model_name="deepseek/deepseek-r1:free",
            provider="openrouter",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            generation_settings=generation_settings,
        ),
    )


def create_ollama_adapter(
    generation_settings: GenerationSettings = GenerationSettings(),
) -> tuple[OpenAICompatibleAdapter, ...]:
    """Create one local Ollama adapter using the shared generation settings."""
    return (
        OpenAICompatibleAdapter(
            model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            provider="ollama",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            generation_settings=generation_settings,
        ),
    )


def _required_environment_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    alternatives = " or ".join(names)
    raise ValueError(
        f"Missing required local API credential: set {alternatives} in the environment."
    )

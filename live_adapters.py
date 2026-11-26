"""REST adapters for HY3, Groq-compatible, and DeepSeek models.

API keys are read only from environment variables. Credentials are never
stored in adapter metadata or response JSONL files.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hallucination_detector.generation import GeneratedText, PromptRecord


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE entries when environment variables are not already set."""
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw_value = stripped.split("=", 1)
        name = name.strip()
        value = raw_value.strip()
        if not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name and not os.getenv(name):
            os.environ[name] = value


_load_dotenv(Path(__file__).resolve().parent / ".env")


class _AdapterError(RuntimeError):
    """Sanitized provider error with no request credentials or response secrets."""


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            raw_detail = error.read(4000).decode("utf-8", errors="replace")
            try:
                parsed_detail = json.loads(raw_detail)
                nested = parsed_detail.get("error")
                if isinstance(nested, dict):
                    detail = nested.get("message") or nested.get("error") or nested.get("detail") or ""
                elif isinstance(nested, str):
                    detail = nested
                else:
                    detail = parsed_detail.get("message") or parsed_detail.get("detail") or ""
            except json.JSONDecodeError:
                detail = raw_detail
        except Exception:
            detail = ""
        for secret_name in (
            "HY3_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "GROK_API_KEY",
            "DEEPSEEK_API_KEY", "ORCAROUTER_API_KEY",
        ):
            secret = os.getenv(secret_name)
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        detail = detail.replace("Authorization", "Authorization [REDACTED]")
        detail = " ".join(detail.split())[:500]
        message = f"Provider returned HTTP {error.code}"
        if detail:
            message += f": {detail}"
        raise _AdapterError(message) from None
    except urllib.error.URLError as error:
        raise _AdapterError("Provider request failed due to network or DNS connectivity.") from None
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise _AdapterError("Provider returned invalid JSON.") from error
    if not isinstance(value, dict):
        raise _AdapterError("Provider JSON response must contain an object.")
    return value


def _timeout() -> float:
    raw = os.getenv("MODEL_API_TIMEOUT_SECONDS", "60")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("MODEL_API_TIMEOUT_SECONDS must be a number.") from error
    if value <= 0:
        raise ValueError("MODEL_API_TIMEOUT_SECONDS must be greater than zero.")
    return value


def _temperature() -> float:
    raw = os.getenv("MODEL_TEMPERATURE", "0")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("MODEL_TEMPERATURE must be a number.") from error
    if not 0 <= value <= 2:
        raise ValueError("MODEL_TEMPERATURE must be between 0 and 2.")
    return value


@dataclass(frozen=True)
class OpenAICompatibleAdapter:
    model_name: str
    api_key: str
    base_url: str
    provider: str
    temperature: float
    timeout: float
    metadata: dict[str, object]

    def generate(self, record: PromptRecord) -> GeneratedText:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": record.prompt}],
            "temperature": self.temperature,
        }
        response = _post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise _AdapterError(f"{self.provider} response did not contain message text.") from error
        if not isinstance(text, str) or not text.strip():
            raise _AdapterError(f"{self.provider} returned empty text.")
        return GeneratedText(text.strip())


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable {name}.")
    return value


def build_adapters() -> tuple[Any, ...]:
    """Build HY3, Groq-compatible, and DeepSeek adapters from environment variables."""
    temperature = _temperature()
    timeout = _timeout()

    hy3_key = os.getenv("OPENROUTER_API_KEY") or _required_env("HY3_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not groq_key:
        raise ValueError("Missing required environment variable GROQ_API_KEY (gsk_ Groq key).")
    deepseek_key = os.getenv("ORCAROUTER_API_KEY") or _required_env("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1")
    deepseek_model = os.getenv("ORCAROUTER_MODEL", "deepseek/deepseek-v4.1-flash-free")

    hy3_model = os.getenv("OPENROUTER_MODEL", "tencent/hy3:free")
    hy3_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return (
        OpenAICompatibleAdapter(
            model_name=hy3_model,
            api_key=hy3_key,
            base_url=hy3_base_url,
            provider="hy3",
            temperature=temperature,
            timeout=timeout,
            metadata={
                "provider": "openrouter-hy3",
                "model_version": hy3_model,
                "research_use": True,
            },
        ),
        OpenAICompatibleAdapter(
            model_name=groq_model,
            api_key=groq_key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            provider="groq",
            temperature=temperature,
            timeout=timeout,
            metadata={
                "provider": "groq",
                "model_version": groq_model,
                "research_use": True,
            },
        ),
        OpenAICompatibleAdapter(
            model_name=deepseek_model,
            api_key=deepseek_key,
            base_url=deepseek_base_url,
            provider="orcarouter-deepseek",
            temperature=temperature,
            timeout=timeout,
            metadata={
                "provider": "orcarouter-deepseek",
                "model_version": deepseek_model,
                "research_use": True,
            },
        ),
    )


def build_deepseek_adapter() -> tuple[Any, ...]:
    """Build only the working OrcaRouter DeepSeek adapter for a single-provider pilot."""
    temperature = _temperature()
    timeout = _timeout()
    key = os.getenv("ORCAROUTER_API_KEY") or _required_env("DEEPSEEK_API_KEY")
    model = os.getenv("ORCAROUTER_MODEL", "deepseek/deepseek-v4.1-flash-free")
    base_url = os.getenv("ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1")
    return (
        OpenAICompatibleAdapter(
            model_name=model,
            api_key=key,
            base_url=base_url,
            provider="orcarouter-deepseek",
            temperature=temperature,
            timeout=timeout,
            metadata={
                "provider": "orcarouter-deepseek",
                "model_version": model,
                "research_use": True,
            },
        ),
    )


def build_ollama_adapter() -> tuple[Any, ...]:
    """Build a local Ollama adapter using its OpenAI-compatible endpoint."""
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    return (
        OpenAICompatibleAdapter(
            model_name=model,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            provider="ollama",
            temperature=_temperature(),
            timeout=_timeout(),
            metadata={
                "provider": "ollama",
                "model_version": model,
                "research_use": False,
            },
        ),
    )


def probe_adapters(adapters: tuple[Any, ...]) -> dict[str, str]:
    """Send one minimal request per adapter; return provider statuses only."""
    statuses: dict[str, str] = {}
    for index, adapter in enumerate(adapters, start=1):
        provider = adapter.metadata.get("provider", f"provider-{index}")
        try:
            result = adapter.generate(PromptRecord(f"probe-{index}", "Reply with exactly one word: ready."))
            statuses[str(provider)] = "PASS" if result.text.strip() else "FAIL empty response"
        except Exception as error:
            statuses[str(provider)] = f"FAIL {type(error).__name__}: {error}"
    return statuses

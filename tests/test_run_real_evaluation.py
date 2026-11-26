from types import SimpleNamespace

from scripts import run_real_evaluation
from hallucination_detector.provider_adapters import GenerationSettings


def test_adapter_factory_receives_generation_settings_when_supported(monkeypatch):
    settings = GenerationSettings(max_output_tokens=256, timeout_seconds=300)
    adapter = object()
    received = {}

    def build_adapters(*, generation_settings):
        received["settings"] = generation_settings
        return (adapter,)

    monkeypatch.setattr(
        run_real_evaluation.importlib,
        "import_module",
        lambda _name: SimpleNamespace(build_adapters=build_adapters),
    )

    adapters = run_real_evaluation._load_adapters(
        "local_adapters:build_adapters", settings
    )

    assert adapters == (adapter,)
    assert received["settings"] == settings


def test_adapter_factory_without_settings_keeps_zero_argument_compatibility(monkeypatch):
    adapter = object()

    def build_adapters():
        return (adapter,)

    monkeypatch.setattr(
        run_real_evaluation.importlib,
        "import_module",
        lambda _name: SimpleNamespace(build_adapters=build_adapters),
    )

    adapters = run_real_evaluation._load_adapters(
        "local_adapters:build_adapters", GenerationSettings()
    )

    assert adapters == (adapter,)

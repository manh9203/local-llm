import pytest

from local_llm.models.registry import (
    DEFAULT_MODEL_ID,
    config_from_env,
    get_model_spec,
)


def test_default_model(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    config = config_from_env()

    assert config.model_id == DEFAULT_MODEL_ID


def test_model_selection(monkeypatch):
    monkeypatch.setenv(
        "LOCAL_LLM_MODEL",
        "mlx-community/Qwen3-4B-4bit",
    )

    config = config_from_env()

    assert config.model_id == (
        "mlx-community/Qwen3-4B-4bit"
    )


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_spec("unknown-model")


def test_model_capabilities():
    spec = get_model_spec(
        "mlx-community/Qwen3-8B-4bit"
    )

    assert spec.backend == "mlx_lm"
    assert spec.family == "qwen3"
    assert spec.enable_prefix_cache

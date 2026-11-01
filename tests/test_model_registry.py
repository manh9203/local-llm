import pytest

from local_llm.config import ChatConfig
from local_llm.models.registry import (
    DEFAULT_MODEL_ID,
    config_from_env,
    get_model_spec,
)


def test_default_model(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    config = config_from_env()

    assert config.model_id == DEFAULT_MODEL_ID == "mlx-community/Qwen3.5-9B-4bit"
    assert ChatConfig().model_id == DEFAULT_MODEL_ID
    assert config.use_cache
    assert not config.enable_thinking


@pytest.mark.parametrize("model_id", [
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3.5-9B-4bit",
])
def test_model_selection(monkeypatch, model_id):
    monkeypatch.setenv(
        "LOCAL_LLM_MODEL",
        model_id,
    )

    config = config_from_env()

    assert config.model_id == model_id


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


def test_cache_strategies():
    assert get_model_spec(DEFAULT_MODEL_ID).cache_strategy == "append_only"
    assert get_model_spec(
        "mlx-community/Qwen3-8B-4bit").cache_strategy == "trimmable"

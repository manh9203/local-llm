import os
from dataclasses import dataclass, replace
from typing import Literal

from local_llm.config import ChatConfig

Backend = Literal["mlx_lm"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: Backend
    family: str
    enable_prefix_cache: bool = True


DEFAULT_MODEL_ID = "mlx-community/Qwen3-8B-4bit"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mlx-community/Qwen3-8B-4bit": ModelSpec(
        model_id="mlx-community/Qwen3-8B-4bit",
        backend="mlx_lm",
        family="qwen3",
    ),
    "mlx-community/Qwen3-4B-4bit": ModelSpec(
        model_id="mlx-community/Qwen3-4B-4bit",
        backend="mlx_lm",
        family="qwen3",
    ),
    "mlx-community/Qwen3.5-9B-4bit": ModelSpec(
        model_id="mlx-community/Qwen3.5-9B-4bit",
        backend="mlx_lm",
        family="qwen3_5",
        enable_prefix_cache=False,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        raise ValueError(
            f"Unsupported model: {model_id}"
        ) from None


def config_from_env() -> ChatConfig:
    model_id = os.environ.get(
        "LOCAL_LLM_MODEL",
        DEFAULT_MODEL_ID,
    )

    spec = get_model_spec(model_id)

    config = ChatConfig()

    return replace(
        config,
        model_id=spec.model_id,
        use_cache=(
            config.use_cache
            and spec.enable_prefix_cache
        ),
    )

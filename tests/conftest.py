"""Explicit opt-ins keep normal pytest runs offline and free of model inference."""

import os
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

MLX_MODULES = {
    "test_inference_engine.py",
    "test_kv_cache.py",
    "test_hybrid_cache_integration.py",
}
OPTIONAL = {
    "mlx": "RUN_MLX_INTEGRATION",
    "tokenizer": "RUN_TOKENIZER_INTEGRATION",
    "docker": "RUN_CODE_INTEGRATION",
    "benchmark": "RUN_BENCHMARKS",
}


def reject_mlx_operation(*args, **kwargs):
    raise AssertionError(
        "Real MLX operations are disabled in CPU tests; "
        "set RUN_MLX_INTEGRATION=1 on Apple Silicon to use the real backend"
    )


class UnavailableKVCache:
    def __init__(self, *args, **kwargs):
        reject_mlx_operation()


# Install explicit test-only sentinels before pytest imports application modules.
# Real integration runs always import the actual MLX packages instead.
if os.environ.get("RUN_MLX_INTEGRATION") != "1":
    for module_name in (
        "mlx",
        "mlx.core",
        "mlx_lm",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_lm.sample_utils",
    ):
        module = ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None)
        sys.modules[module_name] = module
        if "." in module_name:
            parent, _, child = module_name.rpartition(".")
            setattr(sys.modules[parent], child, module)
    sys.modules["mlx_lm"].load = reject_mlx_operation
    sys.modules["mlx_lm"].stream_generate = reject_mlx_operation
    sys.modules["mlx_lm.sample_utils"].make_sampler = reject_mlx_operation
    sys.modules["mlx_lm.models.cache"].KVCache = UnavailableKVCache
    for operation in (
        "make_prompt_cache",
        "can_trim_prompt_cache",
        "trim_prompt_cache",
    ):
        setattr(sys.modules["mlx_lm.models.cache"], operation, reject_mlx_operation)


def pytest_collection_modifyitems(items):
    for item in items:
        filename = item.path.name
        if filename in MLX_MODULES:
            category = "mlx"
        elif filename == "test_chat_template.py":
            category = "tokenizer"
        elif "docker_grader" in item.fixturenames:
            category = "docker"
        elif filename == "test_api.py":
            category = "api"
        else:
            category = "unit"
        item.add_marker(getattr(pytest.mark, category))
        for marker, variable in OPTIONAL.items():
            if item.get_closest_marker(marker) and os.environ.get(variable) != "1":
                item.add_marker(pytest.mark.skip(reason=f"Set {variable}=1 to opt in"))


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch, request):
    # Model-selection tests set their own value; inherited shell state must not
    # change default-model/API assertions or the hardware regression baseline.
    if not request.node.get_closest_marker("mlx"):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

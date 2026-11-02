"""Guard CPU tests against accidentally executing the real inference backend."""

import os

import pytest

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") == "1",
    reason="This test verifies the CPU-only backend guard",
)
def test_cpu_model_loading_fails_before_backend_execution():
    with pytest.raises(AssertionError, match="Real MLX operations are disabled"):
        InferenceEngine(ChatConfig())

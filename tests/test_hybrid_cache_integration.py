import os

import pytest

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="MLX integration disabled",
)
def test_qwen35_cache_clone_independence():
    engine = InferenceEngine(
        ChatConfig(
            model_id="mlx-community/Qwen3.5-9B-4bit",
            max_context_tokens=8192,
            max_new_tokens=64,
            temperature=0,
            enable_thinking=False,
            use_cache=True,
        )
    )

    checkpoint = engine.create_session()

    "".join(checkpoint.stream(
        "My laptop has 16 GB of unified memory. "
        "Acknowledge briefly."
    ))

    assert checkpoint.prompt_cache is not None
    assert checkpoint.cached_tokens

    original_tokens = checkpoint.cached_tokens.copy()
    original_size = checkpoint.cache_size()

    branch_a = checkpoint.clone_cached_session()
    branch_b = checkpoint.clone_cached_session()

    assert branch_a.prompt_cache is not checkpoint.prompt_cache
    assert branch_b.prompt_cache is not checkpoint.prompt_cache

    answer_a = "".join(branch_a.stream(
        "How much unified memory does my laptop have?"
    ))

    assert "16" in answer_a
    assert branch_a.last_stats["reused_tokens"] > 0

    # First branch must not advance the checkpoint.
    assert checkpoint.cached_tokens == original_tokens
    assert checkpoint.cache_size() == original_size

    answer_b = "".join(branch_b.stream(
        "What memory specification did I mention?"
    ))

    assert "16" in answer_b
    assert branch_b.last_stats["reused_tokens"] > 0

    assert checkpoint.cached_tokens == original_tokens
    assert checkpoint.cache_size() == original_size

    # Reuse the checkpoint again after both forks.
    branch_c = checkpoint.clone_cached_session()

    answer_c = "".join(branch_c.stream(
        "Repeat the memory capacity I mentioned."
    ))

    assert "16" in answer_c
    assert branch_c.last_stats["reused_tokens"] > 0

    assert checkpoint.cached_tokens == original_tokens
    assert checkpoint.cache_size() == original_size


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="MLX integration disabled",
)
def test_qwen35_cancelled_branch_preserves_checkpoint():
    engine = InferenceEngine(
        ChatConfig(
            model_id="mlx-community/Qwen3.5-9B-4bit",
            max_new_tokens=128,
            temperature=0,
            enable_thinking=False,
            use_cache=True,
        )
    )

    checkpoint = engine.create_session()

    "".join(checkpoint.stream(
        "Remember that my laptop has 16 GB of memory."
    ))

    original_tokens = checkpoint.cached_tokens.copy()
    original_size = checkpoint.cache_size()

    branch = checkpoint.clone_cached_session()

    stream = branch.stream(
        "Explain how recurrent attention works in detail."
    )

    next(stream)
    stream.close()

    assert branch.prompt_cache is None
    assert branch.cached_tokens == []

    assert checkpoint.cached_tokens == original_tokens
    assert checkpoint.cache_size() == original_size

    another_branch = checkpoint.clone_cached_session()

    answer = "".join(another_branch.stream(
        "How much memory does my laptop have?"
    ))

    assert "16" in answer
    assert another_branch.last_stats["reused_tokens"] > 0

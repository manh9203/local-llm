import os
from dataclasses import replace

import pytest

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine
from local_llm.models.registry import config_from_env


@pytest.fixture(scope="module")
def engine():
    config = replace(
        config_from_env(),
        temperature=0,
        max_new_tokens=64,
    )
    return InferenceEngine(config)


@pytest.mark.skipif(
    os.getenv("RUN_MLX_INTEGRATION") != "1",
    reason="Requires loading an MLX model",
)
def test_sessions_share_model_but_not_state():
    engine = InferenceEngine(
        ChatConfig(
            max_new_tokens=32,
            temperature=0.0,
        )
    )

    session_a = engine.create_session()
    session_b = engine.create_session()

    # Both sessions reference the same loaded model.
    assert session_a.model is session_b.model
    assert session_a.tokenizer is session_b.tokenizer

    # Their conversation histories are separate.
    assert session_a.messages is not session_b.messages
    assert session_a.messages == session_b.messages

    # Run inference only in session A.
    answer = "".join(session_a.stream("What is MLX?"))

    assert answer
    assert len(session_a.messages) == 3
    assert len(session_b.messages) == 1

    # Session A has its own populated KV cache.
    assert session_a.prompt_cache is not None
    assert session_a.cache_size() > 0

    # Session B has not allocated a KV cache yet.
    assert session_b.prompt_cache is None
    assert session_b.cached_tokens == []

    # Clearing session A must not change session B.
    session_a.reset()

    assert len(session_a.messages) == 1
    assert len(session_b.messages) == 1
    assert session_a.prompt_cache is None


@pytest.mark.skipif(
    os.getenv("RUN_MLX_INTEGRATION") != "1",
    reason="Requires loading an MLX model",
)
def test_stream_prepared_matches_normal_stream(engine):
    prompt = "What is a KV cache? Answer briefly."

    session_a = engine.create_session()
    session_b = engine.create_session()

    prepared = session_b.prepare_messages(
        [
            {
                "role": "system",
                "content": engine.config.system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    answer_a = "".join(session_a.stream(prompt))
    answer_b = "".join(session_b.stream_prepared(prepared))

    assert answer_a == answer_b

    assert session_a.last_stats is not None
    assert session_b.last_stats is not None

    assert session_b.last_stats["total_prompt_tokens"] == len(prepared.token_ids)


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="MLX integration tests are disabled",
)
def test_qwen35_uncached_multi_turn():
    engine = InferenceEngine(
        ChatConfig(
            model_id="mlx-community/Qwen3.5-9B-4bit",
            max_context_tokens=8192,
            max_new_tokens=64,
            temperature=0,
            enable_thinking=False,
            use_cache=False,
        )
    )

    session = engine.create_session()

    first = "".join(
        session.stream("My computer has 16 GB of unified memory. Reply briefly.")
    )

    assert first.strip()
    assert session.last_stats["reused_tokens"] == 0
    assert session.prompt_cache is None

    second = "".join(
        session.stream("How much unified memory does my computer have? Answer briefly.")
    )

    assert "16" in second
    assert session.last_stats["reused_tokens"] == 0
    assert session.prompt_cache is None

    assert len(session.messages) == 5

    session.reset()

    assert len(session.messages) == 1
    assert session.prompt_cache is None
    assert session.cached_tokens == []


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_INTEGRATION") != "1",
    reason="MLX integration disabled",
)
def test_qwen35_append_only_multi_turn():
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

    session = engine.create_session()

    assert session.cache_strategy == "append_only"

    answer1 = "".join(
        session.stream(
            "My laptop has 16 GB of unified memory. Acknowledge in one sentence."
        )
    )

    assert answer1.strip()
    assert session.last_stats["reused_tokens"] == 0

    answer2 = "".join(session.stream("How much unified memory does my laptop have?"))

    assert "16" in answer2
    assert session.last_stats["reused_tokens"] > 0

    answer3 = "".join(
        session.stream("What hardware specification did I mention initially?")
    )

    assert answer3.strip()
    assert session.last_stats["reused_tokens"] > 0

    session.reset()

    assert session.prompt_cache is None
    assert session.cached_tokens == []

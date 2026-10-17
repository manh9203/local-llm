
import os

import pytest

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine


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
    answer = "".join(
        session_a.stream("What is MLX?")
    )

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

import os
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, make_prompt_cache

from local_llm.inference.engine import InferenceEngine
from local_llm.models.registry import config_from_env

RUN_INTEGRATION = os.getenv("RUN_MLX_INTEGRATION") == "1"

PROMPTS = [
    "Explain transformer attention in two sentences.",
    "What are query vectors?",
    "How do key vectors relate to queries?",
    "What information do value vectors contain?",
    "Summarize the relationship between Q, K, and V.",
]


@pytest.fixture(scope="module")
def session():
    if not RUN_INTEGRATION:
        pytest.skip("Set RUN_MLX_INTEGRATION=1")

    config = replace(
        config_from_env(),
        temperature=0.0,
        max_new_tokens=128,
    )

    engine = InferenceEngine(config)
    return engine.create_session()


def run_conversation(session, prompts, use_cache):
    """Run a fresh conversation and collect statistics."""
    session.reset()
    session.cache_enabled = use_cache

    outputs = []
    stats = []

    for prompt in prompts:
        list(session.stream(prompt))

        assert session.last_stats is not None

        outputs.append(session.last_generated_tokens.copy())
        stats.append(session.last_stats.copy())

    return outputs, stats


def test_fixed_token_logits(session):
    """Compare full-prefill and prefix-cached logits."""
    prompt = session.tokenize_messages([
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": (
                "Explain the relationship between "
                "queries, keys, and values in attention."
            ),
        },
    ])

    # Split the SAME token sequence into two parts.
    split = len(prompt) // 2

    prefix = prompt[:split]
    suffix = prompt[split:]

    assert prefix
    assert suffix

    model = session.model

    # Path A: process every token together.
    full_cache = make_prompt_cache(model)

    full_logits = model(
        mx.array(prompt)[None],
        cache=full_cache,
    )

    mx.eval(full_logits, [c.state for c in full_cache])

    # Path B: process the prefix first, then the suffix.
    cached = make_prompt_cache(model)

    prefix_logits = model(
        mx.array(prefix)[None],
        cache=cached,
    )

    mx.eval(prefix_logits, [c.state for c in cached])

    # Hybrid recurrent layers have no token-length position.
    attention_caches = [c for c in cached if isinstance(c, KVCache)]
    assert attention_caches
    assert all(c.size() == len(prefix) for c in attention_caches)

    suffix_logits = model(
        mx.array(suffix)[None],
        cache=cached,
    )

    mx.eval(suffix_logits, [c.state for c in cached])

    assert all(c.size() == len(prompt) for c in attention_caches)

    # Compare next-token predictions at the final position.
    a = np.array(
        full_logits[0, -1, :].astype(mx.float32)
    )

    b = np.array(
        suffix_logits[0, -1, :].astype(mx.float32)
    )

    assert np.isfinite(a).all()
    assert np.isfinite(b).all()

    max_diff = float(np.max(np.abs(a - b)))
    mean_diff = float(np.mean(np.abs(a - b)))

    print(f"\nMax absolute logit difference: {max_diff:.6f}")
    print(f"Mean absolute logit difference: {mean_diff:.6f}")

    print(f"Full prefill top token: {np.argmax(a)}")
    print(f"Cached prefill top token: {np.argmax(b)}")

    # Initial diagnostic tolerance, not a universal MLX guarantee.
    np.testing.assert_allclose(
        a,
        b,
        rtol=0.02,
        atol=0.5,
    )


def test_five_turn_cache_invariants(session):
    """Validate reuse and token bookkeeping over five turns."""
    outputs, stats = run_conversation(
        session,
        PROMPTS,
        use_cache=True,
    )

    assert len(outputs) == len(PROMPTS)
    assert len(stats) == len(PROMPTS)

    assert stats[0]["reused_tokens"] == 0

    for turn, s in enumerate(stats):
        assert s["processed_prompt_tokens"] > 0

        assert (
            s["total_prompt_tokens"]
            == s["reused_tokens"]
            + s["processed_prompt_tokens"]
        )

        if turn > 0:
            assert s["reused_tokens"] > 0

        assert s["generation_tokens"] > 0

    # Final cache size must match the tracked token sequence.
    assert session.prompt_cache is not None

    assert session.cache_size() == len(session.cached_tokens)

    # Verify the cache is reusable for the next prompt.
    next_messages, token_ids = session.prepare_prompt(
        "Give a short example of self-attention."
    )

    del next_messages

    input_tokens, reused = session.prepare_cache(token_ids)

    assert reused > 0
    assert input_tokens == token_ids[reused:]
    assert session.cached_tokens == token_ids[:reused]
    assert session.cache_size() == reused


def test_cache_invalidation(session):
    """Clearing the cache must prevent stale reuse."""
    session.reset()
    session.cache_enabled = True

    list(session.stream("What is transformer attention?"))

    assert session.prompt_cache is not None
    assert session.cache_size() > 0

    old_messages = session.messages.copy()

    session.clear_cache()

    assert session.prompt_cache is None
    assert session.cached_tokens == []
    assert session.messages == old_messages

    list(session.stream("What are key vectors?"))

    assert session.last_stats["reused_tokens"] == 0

    assert (
        session.last_stats["processed_prompt_tokens"]
        == session.last_stats["total_prompt_tokens"]
    )

    session.reset()

    assert len(session.messages) == 1
    assert session.prompt_cache is None
    assert session.cached_tokens == []

    list(session.stream("What is transformer attention?"))

    assert session.last_stats["reused_tokens"] == 0


def test_reset_matches_fresh_conversation(session):
    """Resetting history should restore fresh prompt behavior."""
    prompt = "Explain the difference between keys and values."

    session.reset()
    session.cache_enabled = False

    list(session.stream(prompt))
    expected = session.last_generated_tokens.copy()

    session.reset()
    session.cache_enabled = True

    list(session.stream("Explain Python decorators."))
    list(session.stream("Give an example."))

    # Reset both conversation history and KV cache.
    session.reset()
    list(session.stream(prompt))

    assert session.last_generated_tokens == expected
    assert session.last_stats["reused_tokens"] == 0


def test_stream_cancellation_clears_cache(session):
    session.reset()
    session.cache_enabled = True

    stream = session.stream(
        "Explain transformer attention in detail."
    )

    next(stream)
    stream.close()

    assert session.prompt_cache is None
    assert session.cached_tokens == []
    assert len(session.messages) == 1
    assert session.last_stats is None


import os

import pytest

from local_llm.chat import ChatConfig, ChatSession

PROMPTS = [
    "Explain transformer attention in two sentences.",
    "What role do key vectors play in that mechanism?",
]


@pytest.mark.skipif(
    os.getenv("RUN_MLX_INTEGRATION") != "1",
    reason="Requires local MLX model inference",
)
def test_cached_generation_matches_uncached():
    config = ChatConfig(
        max_new_tokens=48,
        temperature=0.0,
    )

    # Load the model only once.
    session = ChatSession(config)

    def run_conversation(use_cache):
        session.reset()
        session.cache_enabled = use_cache

        outputs = []
        stats = []

        for prompt in PROMPTS:
            session.generate(prompt)
            outputs.append(session.last_generated_tokens.copy())
            stats.append(session.last_stats.copy())

        return outputs, stats

    uncached, uncached_stats = run_conversation(False)
    cached, cached_stats = run_conversation(True)

    assert uncached == cached, (
        "Cached generation produced different token IDs."
    )

    # First turn should not reuse previous tokens.
    assert cached_stats[0]["reused_tokens"] == 0

    # Second turn should reuse some history.
    assert cached_stats[1]["reused_tokens"] > 0

    # Cache should reduce the prefill workload.
    assert (
        cached_stats[1]["processed_prompt_tokens"]
        < uncached_stats[1]["processed_prompt_tokens"]
    )

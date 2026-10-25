from benchmarks.compaction_quality.downstream_cases import (
    ALL_CASES,
    TECHNICAL,
)
from benchmarks.compaction_quality.downstream_eval import (
    score_answer,
)
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)

MODEL = "mlx-community/Qwen3-8B-4bit"


def test_all_cases_have_complete_turns():
    for case in ALL_CASES:
        assert len(case.segments) == 3
        assert case.probes

        for segment in case.segments:
            assert len(segment) % 2 == 0

            for i, message in enumerate(segment):
                assert message["role"] == (
                    "user" if i % 2 == 0 else "assistant"
                )


def test_correct_answer_passes():
    probe = TECHNICAL.probes[0]

    result = score_answer(
        "The current context is 8192 tokens, "
        "with 512 reserved for generation.",
        probe,
    )

    assert result["passed"]


def test_outdated_answer_fails():
    probe = TECHNICAL.probes[0]

    result = score_answer(
        "The context window is 4096 tokens.",
        probe,
    )

    assert not result["passed"]


def test_compacted_prompt_preserves_recent_turn():
    history = [{
        "role": "system",
        "content": "Be helpful",
    }]

    for segment in TECHNICAL.segments:
        history.extend(segment)

    manager = ConversationManager()

    # Six completed turns, keeping the last one verbatim.
    covered_count = len(history) - 2

    checkpoint = manager.create(
        messages=history,
        covered_count=covered_count,
        summary="Current context is 8192 tokens.",
        model_id=MODEL,
    )

    followup = history + [{
        "role": "user",
        "content": "What is the context limit?",
    }]

    effective = build_effective_messages(
        followup,
        checkpoint,
        MODEL,
    )

    assert manager.find(followup, MODEL) == checkpoint

    assert "8192" in effective[0]["content"]

    # Last complete turn and latest question remain verbatim.
    assert effective[1:] == followup[covered_count:]
    assert effective[-1]["role"] == "user"

    # The original history is not mutated.
    assert history[0]["content"] == "Be helpful"

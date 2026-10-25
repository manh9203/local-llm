from benchmarks.compaction_quality.cases import (
    CASE,
    FactCheck,
)
from benchmarks.compaction_quality.evaluate import (
    score_summary,
)
from local_llm.conversation.manager import ConversationManager


MODEL = "mlx-community/Qwen3-8B-4bit"


def test_fact_matching_is_case_insensitive():
    checks = [
        FactCheck("model", ("qwen3-8b-4bit",)),
    ]

    result = score_summary(
        "We use Qwen3-8B-4bit.",
        checks,
    )

    assert result[0]["passed"]


def test_missing_fact_is_detected():
    checks = [
        FactCheck("memory", ("16 gb", "16gb")),
    ]

    result = score_summary(
        "The user has an M1 Mac.",
        checks,
    )

    assert not result[0]["passed"]


def test_all_segments_have_complete_turns():
    for segment in CASE.segments:
        assert len(segment) % 2 == 0

        for i, message in enumerate(segment):
            expected = (
                "user" if i % 2 == 0 else "assistant"
            )
            assert message["role"] == expected


def test_rolling_checkpoint_matches_history():
    manager = ConversationManager()
    history = [{
        "role": "system",
        "content": "Be helpful",
    }]

    for round_index, segment in enumerate(
        CASE.segments, start=1
    ):
        history.extend(segment)

        checkpoint = manager.create(
            messages=history,
            covered_count=len(history),
            summary=f"Summary {round_index}",
            model_id=MODEL,
        )

        assert manager.find(history, MODEL) == checkpoint

    assert len(manager) == len(CASE.segments)

import pytest

from local_llm.conversation.checkpoint import (
    create_checkpoint,
)
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)

MODEL = "mlx-community/Qwen3-8B-4bit"


def history():
    return [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "I have an M1 Mac"},
        {"role": "assistant", "content": "Understood"},
        {"role": "user", "content": "I use MLX"},
        {"role": "assistant", "content": "Great"},
        {"role": "user", "content": "What should I optimize?"},
    ]


def test_checkpoint_matches_follow_up():
    messages = history()
    manager = ConversationManager()

    checkpoint = manager.create(
        messages,
        covered_count=3,
        summary="User has an M1 Mac.",
        model_id=MODEL,
    )

    assert manager.find(messages, MODEL) == checkpoint

    next_messages = messages + [
        {"role": "assistant", "content": "Optimize prefill"},
        {"role": "user", "content": "How?"},
    ]

    assert manager.find(next_messages, MODEL) == checkpoint


def test_edited_history_invalidates_checkpoint():
    messages = history()
    manager = ConversationManager()

    manager.create(
        messages,
        covered_count=3,
        summary="User has an M1 Mac.",
        model_id=MODEL,
    )

    edited = [m.copy() for m in messages]
    edited[1]["content"] = "I have an M4 Mac"

    assert manager.find(edited, MODEL) is None


def test_edit_after_checkpoint_remains_valid():
    messages = history()
    manager = ConversationManager()

    checkpoint = manager.create(
        messages,
        covered_count=3,
        summary="User has an M1 Mac.",
        model_id=MODEL,
    )

    edited = [m.copy() for m in messages]
    edited[3]["content"] = "I use PyTorch"

    assert manager.find(edited, MODEL) == checkpoint


def test_longest_matching_checkpoint():
    messages = history()
    manager = ConversationManager()

    earlier = manager.create(
        messages,
        covered_count=3,
        summary="User has an M1 Mac.",
        model_id=MODEL,
    )

    later = manager.create(
        messages,
        covered_count=5,
        summary="User has an M1 Mac and uses MLX.",
        model_id=MODEL,
    )

    assert manager.find(messages, MODEL) == later

    edited = [m.copy() for m in messages]
    edited[3]["content"] = "I use PyTorch"

    assert manager.find(edited, MODEL) == earlier


def test_build_effective_messages():
    messages = history()

    checkpoint = create_checkpoint(
        messages,
        covered_count=3,
        summary="The user has an M1 Mac.",
        model_id=MODEL,
    )

    effective = build_effective_messages(
        messages,
        checkpoint,
        MODEL,
    )

    assert len(effective) == 4
    assert "The user has an M1 Mac." in effective[0]["content"]
    assert effective[1:] == messages[3:]

    # Compaction must not change original history.
    assert messages[0]["content"] == "Be helpful"


def test_invalid_checkpoint_boundary():
    with pytest.raises(ValueError, match="complete"):
        create_checkpoint(
            history(),
            covered_count=2,
            summary="Incomplete turn",
            model_id=MODEL,
        )


def test_wrong_model_is_rejected():
    messages = history()
    checkpoint = create_checkpoint(
        messages,
        covered_count=3,
        summary="User has an M1 Mac.",
        model_id=MODEL,
    )

    with pytest.raises(ValueError, match="does not match"):
        build_effective_messages(
            messages,
            checkpoint,
            "different-model",
        )


def test_lru_eviction():
    manager = ConversationManager(max_checkpoints=1)
    messages = history()

    manager.create(messages, 3, "First summary", MODEL)
    manager.create(messages, 5, "Second summary", MODEL)

    assert len(manager) == 1
    assert manager.find(messages, MODEL).covered_count == 5

import pytest

from local_llm.chat import ChatConfig
from local_llm.inference.context import ContextStats, prepare_context
from local_llm.inference.engine import ChatSession


def test_context_truncation_removes_oldest_exchange():
    session = ChatSession.__new__(ChatSession)

    session.config = ChatConfig(
        max_context_tokens=60,
        max_new_tokens=20,
    )

    session.messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]

    # Fake token counts to test the history-removal policy.
    def fake_tokenize(messages):
        return list(range(len(messages) * 10))

    session.tokenize_messages = fake_tokenize

    pending, token_ids = session.prepare_prompt(
        "Latest question"
    )

    assert len(token_ids) <= 40

    assert pending == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
        {"role": "user", "content": "Latest question"},
    ]

    # Preparing a prompt must not mutate stored history.
    assert session.messages[1]["content"] == "Old question"


def tokenize_messages(messages):
    """Fake tokenizer: one token per character and role marker."""
    tokens = []

    for message in messages:
        tokens.append(0)
        tokens.extend(ord(c) for c in message["content"])

    # Simulate a final assistant generation prefix.
    tokens.append(1)
    return tokens


def prepare(messages, budget):
    return prepare_context(
        messages=messages,
        tokenize_messages=tokenize_messages,
        max_context_tokens=budget + 10,
        max_new_tokens=10,
    )


def test_prompt_fits():
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = prepare(messages, budget=100)

    assert result.messages == messages
    assert result.context_stats.dropped_turns == 0
    assert not result.context_stats.truncated
    assert result.context_stats.effective_tokens == len(
        result.token_ids
    )


def test_drops_oldest_turns_first():
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "A" * 20},
        {"role": "assistant", "content": "B" * 20},
        {"role": "user", "content": "C" * 20},
        {"role": "assistant", "content": "D" * 20},
        {"role": "user", "content": "Latest"},
    ]

    result = prepare(messages, budget=75)

    assert result.messages == [
        messages[0],
        messages[3],
        messages[4],
        messages[5],
    ]
    assert result.context_stats.dropped_turns == 1
    assert result.context_stats.truncated
    assert len(result.token_ids) <= 75


def test_oversized_user_message():
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "X" * 100},
    ]

    with pytest.raises(ValueError, match="context budget"):
        prepare(messages, budget=50)


def test_original_messages_unchanged():
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "A" * 50},
        {"role": "assistant", "content": "B" * 50},
        {"role": "user", "content": "Latest"},
    ]

    original = [m.copy() for m in messages]

    result = prepare(messages, budget=30)

    assert messages == original
    assert result.messages != original
    assert result.messages[-1] == original[-1]


def test_exact_budget():
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    required = len(tokenize_messages(messages))
    result = prepare(messages, budget=required)

    assert len(result.token_ids) == required
    assert not result.context_stats.truncated


def test_without_system_message():
    messages = [
        {"role": "user", "content": "A" * 30},
        {"role": "assistant", "content": "B" * 30},
        {"role": "user", "content": "Latest"},
    ]

    result = prepare(messages, budget=20)

    assert result.messages == [messages[-1]]
    assert result.context_stats.dropped_turns == 1


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "Hello"}],
        [
            {"role": "user", "content": "One"},
            {"role": "user", "content": "Two"},
        ],
        [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ],
    ],
)
def test_invalid_conversation(messages):
    with pytest.raises(ValueError):
        prepare(messages, budget=100)


def test_invalid_budget():
    with pytest.raises(ValueError):
        prepare_context(
            messages=[
                {"role": "user", "content": "Hello"},
            ],
            tokenize_messages=tokenize_messages,
            max_context_tokens=512,
            max_new_tokens=512,
        )


def test_context_stats_properties():
    stats = ContextStats(
        original_tokens=9000,
        effective_tokens=6000,
        input_budget=7680,
        dropped_turns=2,
    )

    assert stats.truncated
    assert stats.removed_tokens == 3000
    assert stats.remaining_tokens == 1680
    assert stats.utilization == pytest.approx(6000 / 7680)


def test_prepared_prompt_never_exceeds_budget():
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
        {"role": "assistant", "content": "D" * 100},
        {"role": "user", "content": "Latest question"},
    ]

    result = prepare(messages, budget=150)
    stats = result.context_stats

    assert stats.effective_tokens <= stats.input_budget
    assert 0 <= stats.utilization <= 1
    assert stats.remaining_tokens >= 0
    assert stats.removed_tokens >= 0
    assert stats.dropped_turns > 0

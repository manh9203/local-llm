from local_llm.conversation.compaction import (
    CompactionConfig,
    plan_compaction,
)
from local_llm.conversation.manager import ConversationManager

MODEL = "mlx-community/Qwen3-8B-4bit"


def tokenize(messages):
    return [
        ord(c)
        for message in messages
        for c in message["content"]
    ]


def make_history(turns, chars_per_message=500):
    messages = [
        {"role": "system", "content": "Be helpful"}
    ]

    for i in range(turns):
        messages.extend([
            {
                "role": "user",
                "content": f"Question {i}: " + "X" * chars_per_message,
            },
            {
                "role": "assistant",
                "content": f"Answer {i}: " + "Y" * chars_per_message,
            },
        ])

    messages.append({
        "role": "user",
        "content": "What next?",
    })

    return messages


def config(**overrides):
    values = {
        "trigger_utilization": 0.85,
        "recent_turns_to_keep": 4,
        "summary_max_tokens": 512,
        "min_turns_to_compact": 2,
        "min_new_tokens_to_compact": 768,
        "min_expected_savings_tokens": 512,
        "summary_overhead_tokens": 64,
    }
    values.update(overrides)
    return CompactionConfig(**values)


def plan(messages, manager=None, policy=None, budget=7680):
    return plan_compaction(
        messages=messages,
        model_id=MODEL,
        manager=manager if manager is not None else ConversationManager(),
        tokenize_messages=tokenize,
        input_budget=budget,
        config=policy if policy is not None else config(),
    )


def test_below_threshold():
    messages = make_history(
        3, chars_per_message=100
    )

    assert plan(messages) is None


def test_large_history_triggers_compaction():
    messages = make_history(
        9, chars_per_message=500
    )

    result = plan(messages)

    assert result is not None
    assert result.covered_count == 11
    assert result.new_segment_tokens >= 768
    assert result.expected_savings_tokens >= 512


def test_short_history_not_worth_compacting():
    messages = make_history(
        9, chars_per_message=40
    )

    assert plan(
        messages,
        budget=500,
    ) is None


def test_requires_minimum_new_turns():
    messages = make_history(
        7, chars_per_message=500
    )

    manager = ConversationManager()

    # Simulate a previously successful compaction
    # covering three complete turns.
    manager.create(
        messages,
        covered_count=7,
        summary="Existing summary",
        model_id=MODEL,
    )

    # With 7 completed turns and 4 retained turns,
    # there are no newly eligible turns.
    assert plan(messages, manager) is None


def test_incremental_compaction_requires_growth():
    base = make_history(
        7, chars_per_message=500
    )
    manager = ConversationManager()

    manager.create(
        base,
        covered_count=7,
        summary="Existing summary",
        model_id=MODEL,
    )

    def extend(extra_turns):
        messages = base[:-1].copy()

        for i in range(extra_turns):
            messages.extend([
                {
                    "role": "user",
                    "content": "N" * 500,
                },
                {
                    "role": "assistant",
                    "content": "R" * 500,
                },
            ])

        messages.append(base[-1])
        return messages

    # One new complete turn cannot justify compaction.
    assert plan(
        extend(1), manager, budget=2000
    ) is None

    # Three new complete turns provide enough savings.
    result = plan(
        extend(3), manager, budget=2000
    )

    assert result is not None
    assert result.previous_summary == "Existing summary"
    assert result.covered_count == 13
    assert len(result.messages_to_summarize) == 6

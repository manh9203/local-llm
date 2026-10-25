from local_llm.conversation.compaction import CompactionConfig
from local_llm.conversation.manager import ConversationManager
from local_llm.conversation.service import CompactionService

MODEL = "mlx-community/Qwen3-8B-4bit"


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, messages, previous_summary, max_tokens):
        self.calls.append({
            "messages": [m.copy() for m in messages],
            "previous_summary": previous_summary,
            "max_tokens": max_tokens,
        })

        return f"Summary version {len(self.calls)}"


class FailingSummarizer:
    def summarize(self, messages, previous_summary, max_tokens):
        raise RuntimeError("Simulated inference failure")


def tokenize(messages):
    # Deterministic simple tokenizer for testing.
    return [
        ord(c)
        for message in messages
        for c in message["content"]
    ]


def make_history(completed_turns: int):
    result = [{"role": "system", "content": "Be helpful"}]

    for i in range(completed_turns):
        result.extend([
            {"role": "user", "content": f"Question {i}: " + "X" * 40},
            {"role": "assistant", "content": f"Answer {i}: " + "Y" * 40},
        ])

    result.append({
        "role": "user",
        "content": "What should I do next?",
    })

    return result


def make_service(summarizer, manager=None):
    return CompactionService(
        manager=manager if manager is not None else ConversationManager(),
        summarizer=summarizer,
        config=CompactionConfig(
            trigger_utilization=0.70,
            recent_turns_to_keep=2,
            summary_max_tokens=128,
            min_turns_to_compact=2,
            min_new_tokens_to_compact=0,
            min_expected_savings_tokens=0,
            summary_overhead_tokens=0,
        ),
    )


def test_below_threshold_does_not_compact():
    fake = FakeSummarizer()
    service = make_service(fake)
    history = make_history(4)

    result = service.compact_if_needed(
        history, MODEL, tokenize, input_budget=10000
    )

    assert not result.compacted
    assert not fake.calls
    assert result.messages == history


def test_initial_compaction():
    fake = FakeSummarizer()
    service = make_service(fake)
    history = make_history(5)

    result = service.compact_if_needed(
        history, MODEL, tokenize, input_budget=300
    )

    assert result.compacted
    assert result.covered_count == 7
    assert result.summary == "Summary version 1"

    # System + summary + two completed turns + latest user.
    assert len(result.messages) == 6
    assert "Summary version 1" in result.messages[0]["content"]
    assert len(fake.calls) == 1


def test_incremental_compaction():
    fake = FakeSummarizer()
    service = make_service(fake)

    history = make_history(5)
    first = service.compact_if_needed(
        history, MODEL, tokenize, input_budget=300
    )

    assert first.compacted

    # Extend the same original conversation by two full turns.
    updated = history[:-1] + [
        {"role": "user", "content": "New question 1"},
        {"role": "assistant", "content": "New answer 1"},
        {"role": "user", "content": "New question 2"},
        {"role": "assistant", "content": "New answer 2"},
        history[-1],
    ]

    second = service.compact_if_needed(
        updated, MODEL, tokenize, input_budget=300
    )

    assert second.compacted
    assert second.covered_count > first.covered_count

    # Only newly covered messages are summarized.
    assert (
        fake.calls[1]["previous_summary"]
        == first.summary
    )
    assert len(fake.calls[1]["messages"]) == (
        second.covered_count - first.covered_count
    )


def test_failed_compaction_does_not_save_checkpoint():
    manager = ConversationManager()
    service = make_service(FailingSummarizer(), manager)
    history = make_history(5)

    result = service.compact_if_needed(
        history, MODEL, tokenize, input_budget=300
    )

    assert not result.compacted
    assert len(manager) == 0
    assert result.messages == history


def test_rejects_compaction_when_summary_is_longer():
    fake = FakeSummarizer()
    service = make_service(fake)

    messages = make_history(5)

    # Use a summary deliberately larger than the original history.
    def long_summary(messages, previous_summary, max_tokens):
        return "An unnecessarily long summary. " * 200

    fake.summarize = long_summary

    result = service.compact_if_needed(
        messages,
        MODEL,
        tokenize,
        input_budget=300,
    )

    assert not result.compacted
    assert len(service.manager) == 0
    assert result.messages == messages

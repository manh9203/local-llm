from collections.abc import Callable
from dataclasses import dataclass

from local_llm.conversation.checkpoint import Message
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)


@dataclass(frozen=True)
class CompactionConfig:
    trigger_utilization: float = 0.85
    recent_turns_to_keep: int = 4
    summary_max_tokens: int = 512
    min_turns_to_compact: int = 2
    min_new_tokens_to_compact: int = 768
    min_expected_savings_tokens: int = 512
    summary_overhead_tokens: int = 64

    def __post_init__(self):
        if not 0 < self.trigger_utilization <= 1:
            raise ValueError("Invalid compaction trigger")
        if self.recent_turns_to_keep < 1:
            raise ValueError("Must retain at least one recent turn")
        if self.summary_max_tokens <= 0:
            raise ValueError("Summary budget must be positive")
        if self.min_turns_to_compact < 1:
            raise ValueError("Must compact at least one turn")
        if self.min_new_tokens_to_compact < 0:
            raise ValueError("Invalid minimum new-token threshold")
        if self.min_expected_savings_tokens < 0:
            raise ValueError("Invalid minimum savings threshold")
        if self.summary_overhead_tokens < 0:
            raise ValueError("Invalid summary overhead")


@dataclass(frozen=True)
class CompactionPlan:
    covered_count: int
    messages_to_summarize: list[Message]
    previous_summary: str | None
    current_tokens: int
    new_segment_tokens: int
    expected_savings_tokens: int


def plan_compaction(
    messages: list[Message],
    model_id: str,
    manager: ConversationManager,
    tokenize_messages: Callable[[list[Message]], list[int]],
    input_budget: int,
    config: CompactionConfig,
) -> CompactionPlan | None:
    """Return a plan when compaction is justified.

    Assumes normalized alternating text chat messages.
    The final message is the newest user question.
    """
    if input_budget <= 0:
        raise ValueError("Input budget must be positive")
    if not messages:
        raise ValueError("Conversation cannot be empty")

    checkpoint = manager.find(messages, model_id)
    effective = build_effective_messages(
        messages,
        checkpoint,
        model_id,
    )

    # Determine whether current effective history is near budget.
    current_tokens = len(tokenize_messages(effective))
    if current_tokens < input_budget * config.trigger_utilization:
        return None

    offset = int(messages[0]["role"] == "system")
    completed_turns = (len(messages) - offset - 1) // 2

    # We only compact complete user/assistant pairs.
    eligible_turns = completed_turns - config.recent_turns_to_keep
    if eligible_turns < config.min_turns_to_compact:
        return None

    # Cover system + all eligible complete turns.
    covered_count = offset + eligible_turns * 2
    previous_covered = (
        checkpoint.covered_count if checkpoint else offset
    )

    new_message_count = covered_count - previous_covered
    if new_message_count <= 0:
        return None

    new_turns = new_message_count // 2
    if new_turns < config.min_turns_to_compact:
        return None

    new_segment = [
        message.copy()
        for message in messages[
            previous_covered:covered_count
        ]
    ]

    new_segment_tokens = len(tokenize_messages(new_segment))
    if new_segment_tokens < config.min_new_tokens_to_compact:
        return None

    # Estimate savings by measuring what removing the
    # newly covered segment would remove from the
    # effective prompt, then reserve space for a summary.
    remaining = (
        [message.copy() for message in messages[:offset]]
        + [message.copy() for message in messages[covered_count:]]
    )

    if checkpoint is not None:
        existing_prefix = build_effective_messages(
            messages[:previous_covered],
            checkpoint,
            model_id,
        )
        remaining = existing_prefix + remaining[offset:]

    without_new_segment_tokens = len(tokenize_messages(remaining))
    removable_tokens = current_tokens - without_new_segment_tokens

    estimated_added_summary_tokens = (
        config.summary_max_tokens
        + config.summary_overhead_tokens
    )

    # Conservative assumption: a new summary might
    # consume the entire allowed output budget.
    expected_savings = removable_tokens - estimated_added_summary_tokens
    if expected_savings < config.min_expected_savings_tokens:
        return None

    return CompactionPlan(
        covered_count=covered_count,
        messages_to_summarize=new_segment,
        previous_summary=(
            checkpoint.summary if checkpoint else None
        ),
        current_tokens=current_tokens,
        new_segment_tokens=new_segment_tokens,
        expected_savings_tokens=expected_savings,
    )

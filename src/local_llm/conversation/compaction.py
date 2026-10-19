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

    def __post_init__(self):
        if not 0 < self.trigger_utilization <= 1:
            raise ValueError("Invalid compaction trigger")
        if self.recent_turns_to_keep < 1:
            raise ValueError("Must retain at least one recent turn")
        if self.summary_max_tokens <= 0:
            raise ValueError("Summary budget must be positive")
        if self.min_turns_to_compact < 1:
            raise ValueError("Must compact at least one turn")


@dataclass(frozen=True)
class CompactionPlan:
    covered_count: int
    messages_to_summarize: list[Message]
    previous_summary: str | None


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

    checkpoint = manager.find(messages, model_id)
    effective = build_effective_messages(
        messages,
        checkpoint,
        model_id,
    )

    # Determine whether current effective history is near budget.
    current_tokens = len(tokenize_messages(effective))
    if current_tokens / input_budget < config.trigger_utilization:
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
    if covered_count <= previous_covered:
        return None

    return CompactionPlan(
        covered_count=covered_count,
        messages_to_summarize=[
            message.copy()
            for message in messages[
                previous_covered:covered_count
            ]
        ],
        previous_summary=(
            checkpoint.summary if checkpoint else None
        ),
    )

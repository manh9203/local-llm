from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextStats:
    original_tokens: int
    effective_tokens: int
    input_budget: int
    dropped_turns: int

    @property
    def truncated(self) -> bool:
        return self.dropped_turns > 0

    @property
    def utilization(self) -> float:
        return self.effective_tokens / self.input_budget

    @property
    def removed_tokens(self) -> int:
        return self.original_tokens - self.effective_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.input_budget - self.effective_tokens


@dataclass(frozen=True)
class PreparedPrompt:
    messages: list[dict]
    token_ids: list[int]
    context_stats: ContextStats


def prepare_context(
    messages: list[dict],
    tokenize_messages: Callable[[list[dict]], list[int]],
    max_context_tokens: int,
    max_new_tokens: int,
) -> PreparedPrompt:
    """Prepare a conversation for inference within a token budget.

    Assumes:
      - An optional leading system message
      - Alternating user/assistant messages
      - A final user message awaiting a response

    Removes oldest complete user/assistant turns if needed.
    Never mutates the input messages.
    """
    if not 0 < max_new_tokens < max_context_tokens:
        raise ValueError(
            "Expected 0 < max_new_tokens < max_context_tokens"
        )

    if not messages:
        raise ValueError("Conversation cannot be empty")

    # Validate the shape before attempting truncation.
    offset = int(messages[0]["role"] == "system")

    if messages[-1]["role"] != "user":
        raise ValueError("Last message must have role 'user'")

    for i in range(offset, len(messages)):
        expected_role = (
            "user" if (i - offset) % 2 == 0
            else "assistant"
        )
        if messages[i]["role"] != expected_role:
            raise ValueError(
                f"Invalid role at index {i}: "
                f"expected '{expected_role}'"
            )

    budget = max_context_tokens - max_new_tokens

    # Copy the messages so the caller's history is unchanged.
    pending = [message.copy() for message in messages]

    token_ids = tokenize_messages(pending)
    original_tokens = len(token_ids)
    dropped_turns = 0

    # Ensure the minimum possible prompt fits.
    minimum = (
        [pending[0], pending[-1]]
        if offset
        else [pending[-1]]
    )

    if len(tokenize_messages(minimum)) > budget:
        raise ValueError(
            "System prompt and latest user message "
            "exceed the context budget"
        )

    # Remove oldest complete turns until the prompt fits.
    while len(token_ids) > budget:
        if len(pending) == offset + 1:
            raise RuntimeError(
                "Unable to reduce prompt below context budget"
            )

        del pending[offset: offset + 2]
        dropped_turns += 1
        token_ids = tokenize_messages(pending)

    stats = ContextStats(
        original_tokens=original_tokens,
        effective_tokens=len(token_ids),
        input_budget=budget,
        dropped_turns=dropped_turns,
    )

    return PreparedPrompt(
        messages=pending,
        token_ids=token_ids,
        context_stats=stats,
    )

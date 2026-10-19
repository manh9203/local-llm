from collections import OrderedDict

from local_llm.conversation.checkpoint import (
    CompactionCheckpoint,
    Message,
    create_checkpoint,
)


CheckpointKey = tuple[str, int, str, int]


def checkpoint_key(
    checkpoint: CompactionCheckpoint,
) -> CheckpointKey:
    return (
        checkpoint.model_id,
        checkpoint.covered_count,
        checkpoint.covered_hash,
        checkpoint.version,
    )


class ConversationManager:
    """In-memory LRU store of compaction checkpoints."""

    def __init__(self, max_checkpoints: int = 32):
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")

        self.max_checkpoints = max_checkpoints
        self._checkpoints: OrderedDict[
            CheckpointKey, CompactionCheckpoint
        ] = OrderedDict()

    def save(self, checkpoint: CompactionCheckpoint) -> None:
        key = checkpoint_key(checkpoint)

        self._checkpoints[key] = checkpoint
        self._checkpoints.move_to_end(key)

        while len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.popitem(last=False)

    def create(
        self,
        messages: list[Message],
        covered_count: int,
        summary: str,
        model_id: str,
    ) -> CompactionCheckpoint:
        checkpoint = create_checkpoint(
            messages=messages,
            covered_count=covered_count,
            summary=summary,
            model_id=model_id,
        )
        self.save(checkpoint)
        return checkpoint

    def find(
        self,
        messages: list[Message],
        model_id: str,
    ) -> CompactionCheckpoint | None:
        """Find the matching checkpoint covering most messages."""
        best_key = None
        best_checkpoint = None

        for key, checkpoint in self._checkpoints.items():
            if not checkpoint.matches(messages, model_id):
                continue

            if (
                best_checkpoint is None
                or checkpoint.covered_count
                > best_checkpoint.covered_count
            ):
                best_key = key
                best_checkpoint = checkpoint

        if best_key is not None:
            self._checkpoints.move_to_end(best_key)

        return best_checkpoint

    def clear(self) -> None:
        self._checkpoints.clear()

    def __len__(self) -> int:
        return len(self._checkpoints)


def build_effective_messages(
    messages: list[Message],
    checkpoint: CompactionCheckpoint | None,
    model_id: str,
) -> list[Message]:
    """Replace a covered conversation prefix with its summary."""
    if checkpoint is None:
        return [message.copy() for message in messages]

    if not checkpoint.matches(messages, model_id):
        raise ValueError("Checkpoint does not match conversation")

    if not messages:
        raise ValueError("Conversation cannot be empty")

    offset = int(messages[0]["role"] == "system")

    if checkpoint.covered_count < offset:
        raise ValueError("Checkpoint doesn't cover system prefix")

    if offset:
        system = messages[0].copy()
        original_system_content = system["content"]
    else:
        system = {"role": "system", "content": ""}
        original_system_content = ""

    summary_block = (
        "[Previous conversation summary]\n"
        + checkpoint.summary
        + "\n[/Previous conversation summary]"
    )

    system["content"] = (
        original_system_content + "\n\n" + summary_block
    ).strip()

    # Everything after the covered prefix is preserved verbatim.
    remaining = [
        message.copy()
        for message in messages[checkpoint.covered_count:]
    ]

    return [system] + remaining

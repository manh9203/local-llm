import hashlib
import json
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]

COMPACTION_VERSION = 1


def history_fingerprint(messages: list[Message]) -> str:
    """Stable digest of an ordered sequence of messages."""
    serialized = json.dumps(
        messages,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompactionCheckpoint:
    summary: str
    covered_count: int
    covered_hash: str
    model_id: str
    version: int = COMPACTION_VERSION

    def matches(
        self,
        messages: list[Message],
        model_id: str,
    ) -> bool:
        if self.version != COMPACTION_VERSION:
            return False

        if self.model_id != model_id:
            return False

        if len(messages) < self.covered_count:
            return False

        return (
            history_fingerprint(messages[:self.covered_count])
            == self.covered_hash
        )


def create_checkpoint(
    messages: list[Message],
    covered_count: int,
    summary: str,
    model_id: str,
) -> CompactionCheckpoint:
    if not 0 < covered_count <= len(messages):
        raise ValueError("Invalid covered message count")

    if not summary.strip():
        raise ValueError("Compaction summary cannot be empty")

    if not model_id.strip():
        raise ValueError("Model ID cannot be empty")

    offset = int(messages[0]["role"] == "system")

    if (covered_count - offset) % 2 != 0:
        raise ValueError(
            "Checkpoint must end after a complete user/assistant turn"
        )

    for i in range(offset, covered_count):
        expected = "user" if (i - offset) % 2 == 0 else "assistant"
        if messages[i]["role"] != expected:
            raise ValueError(
                f"Invalid message role at index {i}"
            )

    return CompactionCheckpoint(
        summary=summary,
        covered_count=covered_count,
        covered_hash=history_fingerprint(
            messages[:covered_count]
        ),
        model_id=model_id,
    )

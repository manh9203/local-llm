import logging
from collections.abc import Callable
from dataclasses import dataclass

from local_llm.conversation.checkpoint import Message, create_checkpoint
from local_llm.conversation.compaction import (
    CompactionConfig,
    plan_compaction,
)
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)
from local_llm.conversation.summarizer import Summarizer

logger = logging.getLogger("local_llm.conversation")


@dataclass(frozen=True)
class CompactionResult:
    messages: list[Message]
    compacted: bool
    covered_count: int
    summary: str | None


class CompactionService:
    def __init__(
        self,
        manager: ConversationManager,
        summarizer: Summarizer,
        config: CompactionConfig,
    ):
        self.manager = manager
        self.summarizer = summarizer
        self.config = config

    def compact_if_needed(
        self,
        messages: list[Message],
        model_id: str,
        tokenize_messages: Callable[[list[Message]], list[int]],
        input_budget: int,
    ) -> CompactionResult:
        checkpoint = self.manager.find(messages, model_id)

        plan = plan_compaction(
            messages=messages,
            model_id=model_id,
            manager=self.manager,
            tokenize_messages=tokenize_messages,
            input_budget=input_budget,
            config=self.config,
        )

        if plan is not None:
            try:
                summary = self.summarizer.summarize(
                    messages=plan.messages_to_summarize,
                    previous_summary=plan.previous_summary,
                    max_tokens=self.config.summary_max_tokens,
                )
                if not summary.strip():
                    raise ValueError("Empty summary")

                candidate_checkpoint = create_checkpoint(
                    messages=messages,
                    covered_count=plan.covered_count,
                    summary=summary,
                    model_id=model_id,
                )

                candidate_messages = build_effective_messages(
                    messages,
                    candidate_checkpoint,
                    model_id,
                )

                new_token_count = len(tokenize_messages(candidate_messages))
                actual_savings = plan.current_tokens - new_token_count

                if actual_savings < self.config.min_expected_savings_tokens:
                    logger.info(
                        "Compaction rejected: actual_savings=%d minimum=%d",
                        actual_savings,
                        self.config.min_expected_savings_tokens,
                    )
                else:
                    self.manager.save(candidate_checkpoint)
                    checkpoint = candidate_checkpoint

                    logger.info(
                        "Compaction accepted: saved=%d tokens "
                        "covered=%d",
                        actual_savings,
                        checkpoint.covered_count,
                    )

                    return CompactionResult(
                        messages=candidate_messages,
                        compacted=True,
                        covered_count=checkpoint.covered_count,
                        summary=checkpoint.summary,
                    )

            except Exception:
                logger.exception(
                    "Compaction failed; preserving previous checkpoint"
                )

        return CompactionResult(
            messages=build_effective_messages(
                messages, checkpoint, model_id
            ),
            compacted=False,
            covered_count=(
                checkpoint.covered_count if checkpoint else 0
            ),
            summary=checkpoint.summary if checkpoint else None,
        )

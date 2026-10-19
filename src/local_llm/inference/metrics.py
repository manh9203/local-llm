import logging

from local_llm.inference.context import ContextStats

logger = logging.getLogger("local_llm.inference")


def log_context(stats: ContextStats) -> None:
    logger.info(
        "Context: original=%d effective=%d budget=%d "
        "utilization=%.1f%% dropped_turns=%d removed=%d",
        stats.original_tokens,
        stats.effective_tokens,
        stats.input_budget,
        stats.utilization * 100,
        stats.dropped_turns,
        stats.removed_tokens,
    )


def log_inference(stats: dict) -> None:
    logger.info(
        "Inference: reused=%d prefill=%d ttft=%.3fs "
        "decode_tps=%.2f peak_memory_gb=%.3f",
        stats["reused_tokens"],
        stats["processed_prompt_tokens"],
        stats["first_token_seconds"],
        stats["generation_tps"],
        stats["peak_memory_gb"],
    )

from dataclasses import dataclass


@dataclass(frozen=True)
class FactCheck:
    name: str
    # Accept one of several exact phrases.
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    segments: list[list[dict[str, str]]]
    checks_by_round: list[list[FactCheck]]


CASE = EvaluationCase(
    name="local_llm_project",
    segments=[
        # Round 1: facts and requirements
        [
            {
                "role": "user",
                "content": (
                    "I'm developing a local LLM assistant "
                    "on a base Apple M1 with 16 GB unified "
                    "memory. I use mlx-community/Qwen3-8B-4bit "
                    "through MLX-LM. My context window is "
                    "currently 4096 tokens. The project uses "
                    "Python 3.11 and FastAPI."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We should benchmark prefill, decoding, "
                    "KV-cache reuse, and memory consumption. "
                    "The API should preserve OpenAI-compatible "
                    "chat completion endpoints."
                ),
            },
            {
                "role": "user",
                "content": (
                    "One important constraint: only one model "
                    "inference operation should execute at a time. "
                    "We need to preserve the original conversation "
                    "history from Open WebUI."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We'll share one inference lock and treat "
                    "the incoming conversation as authoritative."
                ),
            },
        ],

        # Round 2: correction and engineering decision
        [
            {
                "role": "user",
                "content": (
                    "Correction: I've increased the context "
                    "window from 4096 to 8192 tokens. "
                    "max_new_tokens remains 512. "
                    "The input budget is therefore 7680 tokens."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We'll use the updated 8192-token limit "
                    "and reserve 512 tokens for generation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "We decided to identify compaction checkpoints "
                    "using SHA-256 fingerprints of covered message "
                    "prefixes, rather than relying on conversation "
                    "IDs. Edits to covered history must invalidate "
                    "the corresponding checkpoint."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We'll retain the longest matching checkpoint "
                    "and fall back to an older valid checkpoint "
                    "when a later prefix no longer matches."
                ),
            },
        ],

        # Round 3: remaining requirements and next steps
        [
            {
                "role": "user",
                "content": (
                    "Our current KV-cache retention limit is "
                    "8192 tokens. We need to avoid unnecessary "
                    "compactions because summarization is slow "
                    "on my M1. Keep four recent completed turns "
                    "verbatim whenever possible."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The compaction policy should measure expected "
                    "token savings before generating another summary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The next unresolved tasks are evaluating "
                    "summary factual retention and comparing "
                    "Qwen3.5 models. Do not implement persistent "
                    "memory yet."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We'll evaluate correctness first, then "
                    "consider other models and persistent memory."
                ),
            },
        ],
    ],
    checks_by_round=[
        [
            FactCheck("hardware", ("apple m1", "m1 mac")),
            FactCheck("memory", ("16 gb", "16gb")),
            FactCheck("model", ("qwen3-8b-4bit",)),
            FactCheck("initial_context", ("4096", "4,096")),
            FactCheck("framework", ("fastapi",)),
            FactCheck("serialized_inference", (
                "one inference",
                "inference lock",
                "serialized inference",
            )),
            FactCheck("original_history", ("open webui",)),
        ],
        [
            FactCheck("hardware", ("apple m1", "m1 mac")),
            FactCheck("memory", ("16 gb", "16gb")),
            FactCheck("model", ("qwen3-8b-4bit",)),
            FactCheck("updated_context", ("8192", "8,192")),
            FactCheck("generation_budget", ("512",)),
            FactCheck("input_budget", ("7680", "7,680")),
            FactCheck("checkpoint_hash", ("sha-256", "sha256")),
            FactCheck("history_edits", ("invalidat",)),
        ],
        [
            FactCheck("hardware", ("apple m1", "m1 mac")),
            FactCheck("model", ("qwen3-8b-4bit",)),
            FactCheck("updated_context", ("8192", "8,192")),
            FactCheck("input_budget", ("7680", "7,680")),
            FactCheck("checkpoint_hash", ("sha-256", "sha256")),
            FactCheck("retained_turns", ("four recent", "4 recent")),
            FactCheck("cache_budget", ("8192", "8,192")),
            FactCheck("quality_evaluation", (
                "factual retention",
                "summary correctness",
                "summary quality",
            )),
            FactCheck("future_model", ("qwen3.5",)),
            FactCheck("memory_deferred", (
                "persistent memory",
                "memory persistence",
            )),
        ],
    ],
)

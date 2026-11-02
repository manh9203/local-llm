import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"

CONTEXT = (
    "This local LLM assistant runs on Apple Silicon. "
    "It supports multi-turn conversations, context compaction, "
    "prefix caching, and checkpoint-based conversation forks. "
    "It prioritizes correctness, responsiveness, and memory efficiency. "
)


def memory_snapshot(label):
    result = {
        "stage": label,
        "active_gb": mx.get_active_memory() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }

    if hasattr(mx, "get_cache_memory"):
        result["allocator_cache_gb"] = mx.get_cache_memory() / 1e9

    print(
        f"{label:<30} "
        f"active={result['active_gb']:.3f} GB "
        f"peak={result['peak_gb']:.3f} GB"
    )

    return result


def generate(session, question):
    started = perf_counter()

    answer = "".join(session.stream(question))

    elapsed = perf_counter() - started

    if session.last_stats is None:
        raise RuntimeError("Generation failed")

    return {
        "elapsed_seconds": elapsed,
        "reused_tokens": session.last_stats["reused_tokens"],
        "prompt_tokens": session.last_stats["total_prompt_tokens"],
        "generated_tokens": session.last_stats["generation_tokens"],
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        default="outputs/benchmarks/cache_clone_memory.json",
    )

    args = parser.parse_args()

    if args.repetitions < 1 or args.max_new_tokens < 1:
        parser.error("Token settings must be positive")
    if Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")

    config = ChatConfig(
        model_id=args.model,
        max_context_tokens=8192,
        max_new_tokens=args.max_new_tokens,
        temperature=0,
        enable_thinking=False,
        use_cache=True,
    )

    engine = InferenceEngine(config)

    checkpoint = engine.create_session()

    if checkpoint.cache_strategy != "append_only":
        raise RuntimeError("This benchmark requires append-only caching")

    results = {
        "model": args.model,
        "repetitions": args.repetitions,
        "snapshots": [],
        "operations": {},
    }

    def snapshot(label):
        results["snapshots"].append(memory_snapshot(label))

    print("\n=== Baseline ===")
    snapshot("Model loaded")

    print("\n=== Create checkpoint ===")

    initial = generate(
        checkpoint,
        CONTEXT * args.repetitions + "\nWhat are the main goals of this assistant?",
    )

    results["operations"]["initial_generation"] = initial

    if not checkpoint.cached_tokens:
        raise RuntimeError("Checkpoint cache is empty")

    original_tokens = checkpoint.cached_tokens.copy()
    original_position = checkpoint.cache_size()

    print(f"Checkpoint tokens: {original_position}")
    snapshot("Checkpoint A created")

    print("\n=== Clone checkpoint ===")

    started = perf_counter()
    branch_b = checkpoint.clone_cached_session()
    clone_b_seconds = perf_counter() - started

    snapshot("Clone B created")

    started = perf_counter()
    branch_c = checkpoint.clone_cached_session()
    clone_c_seconds = perf_counter() - started

    snapshot("Clone C created")

    results["operations"]["cloning"] = {
        "clone_b_seconds": clone_b_seconds,
        "clone_c_seconds": clone_c_seconds,
    }

    print(f"Clone B: {clone_b_seconds:.4f}s\nClone C: {clone_c_seconds:.4f}s")

    print("\n=== Advance branches ===")

    result_b = generate(
        branch_b,
        "Explain why KV caching improves response latency.",
    )

    results["operations"]["branch_b"] = result_b
    snapshot("Branch B advanced")

    result_c = generate(
        branch_c,
        "Explain why context compaction is useful.",
    )

    results["operations"]["branch_c"] = result_c
    snapshot("Branch C advanced")

    # Verify the original checkpoint remains unchanged.
    assert checkpoint.cached_tokens == original_tokens
    assert checkpoint.cache_size() == original_position
    assert result_b["reused_tokens"] > 0
    assert result_c["reused_tokens"] > 0

    print("\n=== Release branches ===")

    branch_b.clear_cache()
    branch_c.clear_cache()

    del branch_b, branch_c
    gc.collect()

    snapshot("Branches released")

    checkpoint.clear_cache()
    del checkpoint
    gc.collect()

    snapshot("All checkpoints released")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(json.dumps(results, indent=2) + "\n")

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()

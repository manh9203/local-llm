import argparse
import csv
import gc
from pathlib import Path

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine

MODEL_ID = "mlx-community/Qwen3-8B-4bit"

FILLER = (
    "The system processes requests using a tokenizer, "
    "a model forward pass, attention state, and a "
    "generation loop. "
)


def build_prompt(session, target_tokens: int) -> str:
    """Build a user prompt near the requested token count."""
    system = {
        "role": "system",
        "content": session.config.system_prompt,
    }

    def token_count(n: int) -> int:
        return len(session.tokenize_messages([
            system,
            {"role": "user", "content": FILLER * n},
        ]))

    # Find a repetition count above the target.
    upper = 1

    while token_count(upper) < target_tokens:
        upper *= 2

    lower = 0

    # Find the largest repetition count that fits.
    while lower < upper:
        mid = (lower + upper + 1) // 2

        if token_count(mid) <= target_tokens:
            lower = mid
        else:
            upper = mid - 1

    return FILLER * lower


def measure(session, prepared, label: str) -> dict:
    answer = "".join(session.stream_prepared(prepared))

    if session.last_stats is None:
        raise RuntimeError("Generation returned no statistics")

    stats = session.last_stats

    assert (
        stats["reused_tokens"]
        + stats["processed_prompt_tokens"]
        == stats["total_prompt_tokens"]
    )

    return {
        "case": label,
        "prompt_tokens": stats["total_prompt_tokens"],
        "reused_tokens": stats["reused_tokens"],
        "prefill_tokens": stats["processed_prompt_tokens"],
        "ttft_seconds": stats["first_token_seconds"],
        "decode_tps": stats["generation_tps"],
        "peak_memory_gb": stats["peak_memory_gb"],
        "output_tokens": stats["generation_tokens"],
        "output_chars": len(answer),
    }


def run_case(engine, target_tokens: int) -> list[dict]:
    # Both sessions share model weights.
    warm = engine.create_session()
    cold = engine.create_session()

    prompt = build_prompt(warm, target_tokens)

    initial = warm.prepare_messages([
        {
            "role": "system",
            "content": engine.config.system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ])

    print(
        f"Seed prompt: {len(initial.token_ids)} tokens",
        flush=True,
    )

    # Populate the warm session's KV cache.
    measure(warm, initial, "seed")

    if warm.prompt_cache is None:
        raise RuntimeError("Seed generation did not retain a cache")

    # The warm session now contains the previous answer.
    follow_up_messages = warm.messages + [{
        "role": "user",
        "content": "Summarize the previous discussion briefly.",
    }]

    prepared = warm.prepare_messages(follow_up_messages)

    if prepared.context_stats.truncated:
        raise RuntimeError(
            "Benchmark follow-up was truncated; "
            "reduce the target prompt length."
        )

    # Run the same follow-up with and without cached state.
    # The cold session has no KV cache.
    cold_result = measure(cold, prepared, "cold")
    warm_result = measure(warm, prepared, "warm")

    assert cold_result["prompt_tokens"] == warm_result["prompt_tokens"]
    assert cold_result["reused_tokens"] == 0

    for result in (cold_result, warm_result):
        result["target_tokens"] = target_tokens

    warm.clear_cache()
    cold.clear_cache()
    gc.collect()

    return [cold_result, warm_result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[2000, 4000, 6000, 7500],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/context_results.csv"),
    )
    args = parser.parse_args()

    config = ChatConfig(
        model_id=MODEL_ID,
        max_context_tokens=8192,
        max_new_tokens=64,
        temperature=0.0,
        use_cache=True,
    )

    engine = InferenceEngine(config)
    results = []

    for target in args.targets:
        print(f"\n=== Target: {target} tokens ===", flush=True)

        case_results = run_case(engine, target)
        results.extend(case_results)

        for result in case_results:
            print(result, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

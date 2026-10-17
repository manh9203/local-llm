
import argparse
import contextlib
import io
import statistics
from dataclasses import replace

from local_llm.chat import ChatConfig, ChatSession

PROMPTS = [
    "Explain how transformer attention works in about 100 words.",
    "Can you explain the role of key and value vectors?",
]


def run_conversation(session, use_cache):
    session.reset()
    session.cache_enabled = use_cache

    results = []

    for prompt in PROMPTS:
        # Suppress assistant text while benchmarking.
        with contextlib.redirect_stdout(io.StringIO()):
            session.generate(prompt)

        results.append({
            "stats": session.last_stats.copy(),
            "tokens": session.last_generated_tokens.copy(),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.trials < 1 or args.max_tokens < 1:
        parser.error("Trials and max tokens must be positive.")

    config = replace(
        ChatConfig(),
        temperature=0.0,
        max_new_tokens=args.max_tokens,
    )

    session = ChatSession(config)

    results = {"off": [], "on": []}

    for trial in range(args.trials):
        # Alternate order to reduce warm-up/thermal bias.
        modes = [False, True] if trial % 2 == 0 else [True, False]

        trial_results = {}

        for enabled in modes:
            trial_results[enabled] = run_conversation(
                session, enabled
            )

        off = trial_results[False]
        on = trial_results[True]

        # Verify both paths ran the same conversation.
        assert [x["tokens"] for x in off] == [
            x["tokens"] for x in on
        ], f"Token mismatch in trial {trial + 1}"

        for enabled, key in [(False, "off"), (True, "on")]:
            second_turn = trial_results[enabled][1]["stats"]
            results[key].append(second_turn)

        print(f"Trial {trial + 1}/{args.trials} completed")

    print("\nSecond-turn results (mean):")

    for mode in ("off", "on"):
        samples = results[mode]

        ttft = statistics.mean(
            x["first_token_seconds"] for x in samples
        )

        prefill_tokens = statistics.mean(
            x["processed_prompt_tokens"] for x in samples
        )

        reused = statistics.mean(
            x["reused_tokens"] for x in samples
        )

        decode = statistics.mean(
            x["generation_tps"] for x in samples
        )

        print(f"\nCache {mode}:")
        print(f"  First token: {ttft:.3f}s")
        print(f"  Prefill tokens: {prefill_tokens:.1f}")
        print(f"  Reused tokens: {reused:.1f}")
        print(f"  Decode speed: {decode:.2f} tok/s")

    off_ttft = statistics.mean(
        x["first_token_seconds"] for x in results["off"]
    )

    on_ttft = statistics.mean(
        x["first_token_seconds"] for x in results["on"]
    )

    improvement = (1 - on_ttft / off_ttft) * 100

    print(f"\nTTFT reduction: {improvement:.1f}%")


if __name__ == "__main__":
    main()

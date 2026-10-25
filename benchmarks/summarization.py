import argparse
import csv
import statistics
from pathlib import Path

from local_llm.config import ChatConfig
from local_llm.conversation.summarizer import MLXSummarizer
from local_llm.inference.engine import InferenceEngine

MODEL_ID = "mlx-community/Qwen3-8B-4bit"


def make_messages(start: int, count: int):
    messages = []

    for i in range(start, start + count):
        messages.extend([
            {
                "role": "user",
                "content": (
                    f"Question {i}: We are developing a "
                    "local LLM inference server using MLX "
                    "and Qwen3-8B. The server supports "
                    "token-aware context management and "
                    "cross-request KV-cache reuse. "
                    "Explain how we should improve it."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Answer {i}: We should measure "
                    "prefill latency, decoding throughput, "
                    "memory usage, and cache reuse. "
                    "The conversation manager can "
                    "summarize older turns while "
                    "preserving recent messages."
                ),
            },
        ])

    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=[128, 256, 512],
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/summarization_results.csv"
        ),
    )

    args = parser.parse_args()

    engine = InferenceEngine(
        ChatConfig(
            model_id=MODEL_ID,
            max_context_tokens=8192,
            max_new_tokens=512,
            temperature=0,
            use_cache=True,
        )
    )

    summarizer = MLXSummarizer(engine)

    cases = {
        "initial": {
            "messages": make_messages(0, 8),
            "previous_summary": None,
        },
        "incremental": {
            "messages": make_messages(8, 4),
            "previous_summary": (
                "The user is building a local LLM "
                "assistant with Qwen3-8B and MLX. "
                "The server uses FastAPI, token-aware "
                "context management, and cross-request "
                "KV-cache reuse. The user wants to "
                "improve performance and reliability."
            ),
        },
    }

    results = []

    for case_name, case in cases.items():
        for limit in args.limits:
            for trial in range(1, args.trials + 1):
                print(
                    f"\nCase={case_name} "
                    f"max_tokens={limit} "
                    f"trial={trial}",
                    flush=True,
                )

                status = "success"

                try:
                    summary = summarizer.summarize(
                        messages=case["messages"],
                        previous_summary=case["previous_summary"],
                        max_tokens=limit,
                    )
                except ValueError as exc:
                    status = f"rejected: {exc}"
                    summary = ""

                stats = summarizer.last_stats

                if stats is None:
                    raise RuntimeError(
                        "Summarizer produced no benchmark metrics"
                    )

                row = {
                    "case": case_name,
                    "max_tokens": limit,
                    "trial": trial,
                    "status": status,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "ttft_seconds": stats.first_token_seconds,
                    "prefill_tps": stats.prefill_tps,
                    "decode_tps": stats.decode_tps,
                    "elapsed_seconds": stats.elapsed_seconds,
                    "peak_memory_gb": stats.peak_memory_gb,
                    "finish_reason": stats.finish_reason,
                    "summary_chars": len(summary),
                }

                results.append(row)
                print(row, flush=True)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {args.output}")

    for case_name in cases:
        for limit in args.limits:
            subset = [
                r["elapsed_seconds"]
                for r in results
                if r["case"] == case_name
                and r["max_tokens"] == limit
            ]

            print(
                f"{case_name}, limit={limit}: "
                f"median={statistics.median(subset):.2f}s"
            )


if __name__ == "__main__":
    main()

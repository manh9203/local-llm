"""Controlled cache on/off measurements across prompt lengths."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from platform import python_version
from statistics import median

import mlx.core as mx

from benchmarks.quality import model_metadata, runtime_versions, save_progress
from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine

CONTEXT = (
    "This local assistant runs on Apple Silicon. Its priorities are correctness, "
    "low latency, reliable context retrieval, and predictable memory usage. "
)
FOLLOWUP = (
    "What are the two most important priorities mentioned earlier? Answer briefly."
)
METRICS = (
    "first_token_seconds",
    "prompt_tps",
    "generation_tps",
    "peak_memory_gb",
    "reused_tokens",
    "processed_prompt_tokens",
    "generation_tokens",
)


def build_prompt(session, target: int) -> list[dict]:
    def messages(count):
        return [
            {"role": "system", "content": session.config.system_prompt},
            {
                "role": "user",
                "content": CONTEXT * count + "\nSummarize the priorities.",
            },
        ]

    low, high = 0, 1
    while len(session.tokenize_messages(messages(high))) <= target:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if len(session.tokenize_messages(messages(mid))) <= target:
            low = mid
        else:
            high = mid
    result = messages(low)
    if low == 0 or len(session.tokenize_messages(result)) > target:
        raise ValueError("Target is too small for the benchmark prompt")
    return result


def measure(session, messages: list[dict]) -> dict:
    prepared = session.prepare_messages(messages)
    if prepared.context_stats.truncated or prepared.messages != messages:
        raise RuntimeError("Benchmark prompt was changed or truncated")
    mx.reset_peak_memory()
    answer = "".join(session.stream_prepared(prepared))
    if session.last_stats is None:
        raise RuntimeError("Generation returned no statistics")
    stats = session.last_stats.copy()
    if stats["reused_tokens"] + stats["processed_prompt_tokens"] != len(
        prepared.token_ids
    ):
        raise RuntimeError("Prompt-token accounting mismatch")
    return {
        **stats,
        "answer": answer,
        "prompt_token_ids": prepared.token_ids,
        "output_token_ids": session.last_generated_tokens.copy(),
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def run_conversation(
    engine, messages: list[dict], cache_enabled: bool, progress=None
) -> dict:
    session = engine.create_session()
    session.cache_enabled = cache_enabled
    try:
        initial = measure(session, messages)
        if progress is not None:
            progress("initial", initial)
        continuation = measure(
            session, session.messages + [{"role": "user", "content": FOLLOWUP}]
        )
        if progress is not None:
            progress("continuation", continuation)
        if initial["reused_tokens"] != 0:
            raise RuntimeError("Initial request unexpectedly reused cached state")
        if cache_enabled and continuation["reused_tokens"] <= 0:
            raise RuntimeError("Cached continuation did not reuse its prefix")
        if not cache_enabled and continuation["reused_tokens"] != 0:
            raise RuntimeError("Uncached continuation reused cached state")
        return {"initial": initial, "continuation": continuation}
    finally:
        session.clear_cache()
        gc.collect()
        mx.clear_cache()


def benchmark_target(engine, target: int, trials: int, progress=None) -> dict:
    result = {"target_tokens": target, "samples": [], "medians": {}}
    if progress is not None:
        progress(result)
    messages = build_prompt(engine.create_session(), target)
    samples = result["samples"]
    for trial in range(trials):
        modes = (False, True) if trial % 2 == 0 else (True, False)
        pair = {"trial": trial + 1, "order": [], "verified": False}
        samples.append(pair)
        for enabled in modes:
            mode = "cache_on" if enabled else "cache_off"
            pair["order"].append(mode)
            pair[mode] = {}
            print(
                f"Target {target}: trial {trial + 1}/{trials}, {mode} started",
                flush=True,
            )

            def record_generation(stage, measurement, sample=pair, label=mode):
                sample[label][stage] = measurement
                if progress is not None:
                    progress(result)
                print(
                    f"  {stage}: {measurement['total_prompt_tokens']} tokens, "
                    f"TTFT={measurement['first_token_seconds']:.3f}s, "
                    f"reused={measurement['reused_tokens']}",
                    flush=True,
                )

            run_conversation(engine, messages, enabled, record_generation)
        for stage in ("initial", "continuation"):
            off, on = pair["cache_off"][stage], pair["cache_on"][stage]
            for field in ("prompt_token_ids", "output_token_ids"):
                if off[field] != on[field]:
                    raise RuntimeError(f"Cache on/off {field} differ in {stage}")
        pair["verified"] = True
        if progress is not None:
            progress(result)
        print(
            f"Target {target}: trial {trial + 1}/{trials}, identical tokens verified",
            flush=True,
        )
    medians = {
        mode: {
            stage: {
                metric: median(sample[mode][stage][metric] for sample in samples)
                for metric in METRICS
            }
            for stage in ("initial", "continuation")
        }
        for mode in ("cache_off", "cache_on")
    }
    result["medians"] = medians
    if progress is not None:
        progress(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--targets", type=int, nargs="+", default=[512, 2048, 6000, 7500]
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.max_new_tokens < 8192:
        parser.error(
            "Trials must be positive; --max-new-tokens must be between 1 and 8191"
        )
    if any(
        target <= 0 or target + 2 * args.max_new_tokens + 256 > 8192
        for target in args.targets
    ):
        parser.error(
            "Targets must be positive and leave room for both answers and the follow-up"
        )
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    config = ChatConfig(
        model_id=args.model,
        temperature=0,
        enable_thinking=False,
        max_context_tokens=8192,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    report = {
        "protocol": "performance-v1",
        "config": asdict(config),
        "python_version": python_version(),
        "runtime": runtime_versions(),
        "trials": args.trials,
        "targets": args.targets,
        "run_status": "running",
        "memory_scope": "MLX process allocations, not total system unified memory",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    try:
        engine = InferenceEngine(config)
        report["model_metadata"] = model_metadata(engine)
        # Exercise generation before timed trials; model loading is also excluded.
        warmup = engine.create_session()
        measure(
            warmup,
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": "Reply with OK."},
            ],
        )
        warmup.clear_cache()
        del warmup
        gc.collect()
        mx.clear_cache()
        for target in args.targets:

            def record_target(result, target_tokens=target):
                if (
                    not report["results"]
                    or report["results"][-1]["target_tokens"] != target_tokens
                ):
                    report["results"].append(result)
                save_progress(args.output, report)

            benchmark_target(engine, target, args.trials, record_target)
        report["run_status"] = "completed"
    except BaseException as error:
        report["run_status"] = "aborted"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save_progress(args.output, report)
        raise
    save_progress(args.output, report)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

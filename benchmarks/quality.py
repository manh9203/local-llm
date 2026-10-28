import argparse
import json
from dataclasses import asdict
from pathlib import Path
from platform import python_version
from time import perf_counter

from benchmarks.quality_cases import ALL_CASES
from benchmarks.scoring import (
    SCORING_VERSION,
    score_run,
)
from local_llm.config import ChatConfig
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)
from local_llm.conversation.summarizer import MLXSummarizer
from local_llm.inference.engine import InferenceEngine

# Retain one completed turn for this small stress test.
RECENT_TURNS_TO_KEEP = 1


def generate_answer(engine, messages: list[dict]) -> dict:
    session = engine.create_session()
    prepared = session.prepare_messages(messages)

    # Full-history comparison must really include full history.
    if prepared.context_stats.truncated:
        raise RuntimeError(
            "Evaluation prompt was truncated; "
            "reduce case size or increase context."
        )

    started = perf_counter()
    answer = "".join(session.stream_prepared(prepared))
    elapsed = perf_counter() - started

    stats = session.last_stats
    if stats is None:
        raise RuntimeError("Generation returned no statistics")

    session.clear_cache()

    return {
        "answer": answer.strip(),
        "prompt_tokens": len(prepared.token_ids),
        "elapsed_seconds": elapsed,
        "ttft_seconds": stats.get("first_token_seconds"),
        "generated_tokens": stats["generation_tokens"],
        "finish_reason": stats["finish_reason"],
        "peak_memory_gb": stats["peak_memory_gb"],
    }


def run_case(engine, case, max_summary_tokens, max_probes, model_id):
    manager = ConversationManager()
    summarizer = MLXSummarizer(engine)

    history = [{
        "role": "system",
        "content": engine.config.system_prompt,
    }]

    previous_summary = None
    previous_covered = 1
    rounds = []
    checkpoint = None

    for round_number, segment in enumerate(
        case.segments, start=1
    ):
        history.extend([message.copy() for message in segment])

        completed_turns = (len(history) - 1) // 2
        covered_count = (
            1 + 2 * (
                completed_turns - RECENT_TURNS_TO_KEEP
            )
        )

        # Only newly eligible turns are summarized.
        new_segment = [
            message.copy()
            for message in history[
                previous_covered:covered_count
            ]
        ]

        started = perf_counter()

        summary = summarizer.summarize(
            messages=new_segment,
            previous_summary=previous_summary,
            max_tokens=max_summary_tokens,
        )

        elapsed = perf_counter() - started

        checkpoint = manager.create(
            messages=history,
            covered_count=covered_count,
            summary=summary,
            model_id=model_id,
        )

        assert checkpoint.matches(history, model_id)

        rounds.append({
            "round": round_number,
            "covered_count": covered_count,
            "summary": summary,
            "summary_seconds": elapsed,
            "stats": (
                asdict(summarizer.last_stats)
                if summarizer.last_stats is not None
                else None
            ),
        })

        print(
            f"Round {round_number}: "
            f"covered={covered_count}, "
            f"summary_time={elapsed:.2f}s",
            flush=True,
        )

        previous_summary = summary
        previous_covered = covered_count

    probes = case.probes[:max_probes]
    evaluations = []

    for probe in probes:
        question = {
            "role": "user",
            "content": (
                probe.question
                + "\nAnswer based only on the conversation. Return only a JSON "
                  "object with exactly the fields and types/choices below. "
                  "Use null or unknown for unspecified information when allowed. "
                  "Do not include explanations.\n"
                + json.dumps(probe.answer_format)
            ),
        }

        full_messages = history + [question]

        # Checkpoint must match the original client history.
        assert manager.find(full_messages, model_id) == checkpoint

        compacted_messages = build_effective_messages(
            full_messages,
            checkpoint,
            model_id,
        )

        full = generate_answer(engine, full_messages)
        compacted = generate_answer(engine, compacted_messages)

        evaluations.append({
            "probe": probe.name,
            "question": question["content"],
            "expected": probe.expected,
            "answer_format": probe.answer_format,
            "full": full,
            "compacted": compacted,
        })

        print(
            f"  {probe.name}: "
            f"tokens={full['prompt_tokens']}"
            f"->{compacted['prompt_tokens']}",
            flush=True,
        )

    return {
        "case": case.name,
        "history": history,
        "rounds": rounds,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-8B-4bit",
    )
    parser.add_argument(
        "--case",
        choices=["all"] + [c.name for c in ALL_CASES],
        default="all",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--summary-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.max_probes is not None and args.max_probes <= 0:
        parser.error("--max-probes must be positive")
    if not 0 < args.summary_tokens < 8192:
        parser.error("--summary-tokens must be between 1 and 8191")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    selected_cases = [
        case for case in ALL_CASES
        if args.case == "all" or case.name == args.case
    ]

    engine = InferenceEngine(
        ChatConfig(
            model_id=args.model,
            max_context_tokens=8192,
            max_new_tokens=128,
            temperature=0,
            enable_thinking=False,
            use_cache=False,
        )
    )

    results = []

    for case in selected_cases:
        print(f"\n=== {case.name} ===", flush=True)

        result = run_case(
            engine,
            case,
            args.summary_tokens,
            args.max_probes,
            args.model
        )

        results.append(result)

    report = {
        "protocol": SCORING_VERSION,
        "model": args.model,
        "config": asdict(engine.config),
        "python_version": python_version(),
        "summary_tokens": args.summary_tokens,
        "recent_turns_to_keep": RECENT_TURNS_TO_KEEP,
        "results": results,
    }
    scores = score_run(report)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\nFull history: {scores['full']}")
    print(f"Compacted: {scores['compacted']}")
    print(f"Confirmed regressions: {scores['confirmed_regressions']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

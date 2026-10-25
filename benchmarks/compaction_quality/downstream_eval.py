import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from benchmarks.compaction_quality.downstream_cases import (
    ALL_CASES,
    Probe,
)
from local_llm.config import ChatConfig
from local_llm.conversation.manager import (
    ConversationManager,
    build_effective_messages,
)
from local_llm.conversation.summarizer import MLXSummarizer
from local_llm.inference.engine import InferenceEngine


MODEL_ID = "mlx-community/Qwen3-8B-4bit"

# Retain one completed turn for this small stress test.
RECENT_TURNS_TO_KEEP = 1


def score_answer(answer: str, probe: Probe) -> dict:
    missing = []

    for alternatives in probe.required:
        matched = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in alternatives
        )
        if not matched:
            missing.append(list(alternatives))

    return {
        "passed": not missing,
        "missing_patterns": missing,
    }


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

    stats = session.last_stats or {}

    session.clear_cache()

    return {
        "answer": answer.strip(),
        "prompt_tokens": len(prepared.token_ids),
        "elapsed_seconds": elapsed,
        "ttft_seconds": stats.get("first_token_seconds"),
    }


def run_case(engine, case, max_summary_tokens, max_probes):
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
            model_id=MODEL_ID,
        )

        assert checkpoint.matches(history, MODEL_ID)

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
                + "\nAnswer briefly based only on the "
                  "conversation. If something was not "
                  "specified, say so."
            ),
        }

        full_messages = history + [question]

        # Checkpoint must match the original client history.
        assert manager.find(full_messages, MODEL_ID) == checkpoint

        compacted_messages = build_effective_messages(
            full_messages,
            checkpoint,
            MODEL_ID,
        )

        full = generate_answer(engine, full_messages)
        compacted = generate_answer(engine, compacted_messages)

        full_score = score_answer(full["answer"], probe)
        compacted_score = score_answer(
            compacted["answer"], probe
        )

        evaluations.append({
            "probe": probe.name,
            "question": probe.question,
            "reference": probe.reference,
            "full": {
                **full,
                **full_score,
            },
            "compacted": {
                **compacted,
                **compacted_score,
            },
        })

        print(
            f"  {probe.name}: "
            f"full={full_score['passed']} "
            f"compacted={compacted_score['passed']} "
            f"tokens={full['prompt_tokens']}"
            f"->{compacted['prompt_tokens']}",
            flush=True,
        )

    return {
        "case": case.name,
        "rounds": rounds,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser()

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
        default=Path(
            "benchmarks/compaction_quality/downstream_results.json"
        ),
    )

    args = parser.parse_args()

    if args.max_probes is not None and args.max_probes <= 0:
        parser.error("--max-probes must be positive")

    selected_cases = [
        case for case in ALL_CASES
        if args.case == "all" or case.name == args.case
    ]

    engine = InferenceEngine(
        ChatConfig(
            model_id=MODEL_ID,
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
        )

        results.append(result)

    all_evaluations = [
        item
        for result in results
        for item in result["evaluations"]
    ]

    full_correct = sum(
        item["full"]["passed"]
        for item in all_evaluations
    )

    compacted_correct = sum(
        item["compacted"]["passed"]
        for item in all_evaluations
    )

    regressions = sum(
        item["full"]["passed"]
        and not item["compacted"]["passed"]
        for item in all_evaluations
    )

    report = {
        "model": MODEL_ID,
        "summary_tokens": args.summary_tokens,
        "total_probes": len(all_evaluations),
        "full_correct": full_correct,
        "compacted_correct": compacted_correct,
        "regressions": regressions,
        "results": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print(
        f"Full history: {full_correct}/{len(all_evaluations)}"
    )
    print(
        f"Compacted: {compacted_correct}/{len(all_evaluations)}"
    )
    print(f"Regressions: {regressions}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

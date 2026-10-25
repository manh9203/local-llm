import argparse
import json
import re
from pathlib import Path

from benchmarks.compaction_quality.cases import CASE
from local_llm.config import ChatConfig
from local_llm.conversation.manager import ConversationManager
from local_llm.conversation.summarizer import MLXSummarizer
from local_llm.inference.engine import InferenceEngine


MODEL_ID = "mlx-community/Qwen3-8B-4bit"


def normalize(text: str) -> str:
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def score_summary(summary, checks):
    normalized = normalize(summary)
    results = []

    for check in checks:
        passed = any(
            normalize(phrase) in normalized
            for phrase in check.alternatives
        )

        results.append({
            "fact": check.name,
            "passed": passed,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/compaction_quality/results.json"
        ),
    )
    args = parser.parse_args()

    engine = InferenceEngine(
        ChatConfig(
            model_id=MODEL_ID,
            max_context_tokens=8192,
            max_new_tokens=512,
            temperature=0,
            enable_thinking=False,
        )
    )

    summarizer = MLXSummarizer(engine)
    manager = ConversationManager()

    original_history = [
        {"role": "system", "content": (
            "You are a helpful local AI assistant."
        )}
    ]

    previous_summary = None
    results = []

    for round_index, segment in enumerate(
        CASE.segments, start=1
    ):
        original_history.extend(segment)

        print(f"\n=== Compaction round {round_index} ===")

        summary = summarizer.summarize(
            messages=segment,
            previous_summary=previous_summary,
            max_tokens=args.max_tokens,
        )

        checkpoint = manager.create(
            messages=original_history,
            covered_count=len(original_history),
            summary=summary,
            model_id=MODEL_ID,
        )

        assert checkpoint.matches(
            original_history,
            MODEL_ID,
        )

        assert manager.find(
            original_history,
            MODEL_ID,
        ) == checkpoint

        checks = score_summary(
            summary,
            CASE.checks_by_round[round_index - 1],
        )

        passed = sum(
            item["passed"] for item in checks
        )

        total = len(checks)

        print(f"Fact recall: {passed}/{total}")
        print(f"Summary:\n{summary}\n")

        stats = summarizer.last_stats

        results.append({
            "round": round_index,
            "summary": summary,
            "covered_count": checkpoint.covered_count,
            "covered_hash": checkpoint.covered_hash,
            "checks": checks,
            "passed": passed,
            "total": total,
            "stats": (
                vars(stats) if stats is not None else None
            ),
        })

        previous_summary = summary

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            {
                "case": CASE.name,
                "model": MODEL_ID,
                "max_tokens": args.max_tokens,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

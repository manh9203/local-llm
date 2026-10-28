"""Score structured quality answers, or rescore saved runs without MLX.

Usage: python -m benchmarks.scoring RUN.json [RUN.json ...]
Each input gets a small sibling .scores.json report; inputs are never rewritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

SCORING_VERSION = "structured-v1"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def score_answer(answer: str, expected: dict) -> dict:
    """Check explicit field values and types, including polarity and unknowns.

    This measures structured retrieval and output compliance, not free-form
    writing quality. A single JSON code fence is tolerated, but no prose.
    """
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string")
    text = answer.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    try:
        actual = json.loads(text, object_pairs_hook=unique_object)
    except ValueError:
        return {"status": "failed", "reason": "invalid_json"}
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return {"status": "failed", "reason": "wrong_fields"}
    wrong_types = [
        key for key, value in expected.items()
        if type(actual[key]) is not type(value)
    ]
    if wrong_types:
        return {"status": "failed", "reason": "wrong_types", "fields": wrong_types}
    mismatches = [
        key for key, value in expected.items()
        if (
            actual[key].strip().casefold() != value.strip().casefold()
            if isinstance(value, str) else actual[key] != value
        )
    ]
    if mismatches:
        return {"status": "failed", "reason": "wrong_values", "fields": mismatches}
    return {"status": "passed"}


def summarize_evaluations(evaluations: list[dict]) -> dict:
    summary = {"total_probes": len(evaluations)}
    for mode in ("full", "compacted"):
        summary[mode] = {
            status: sum(item[mode]["status"] == status for item in evaluations)
            for status in ("passed", "failed", "review")
        }
    summary["confirmed_regressions"] = sum(
        item["full"]["status"] == "passed"
        and item["compacted"]["status"] == "failed"
        for item in evaluations
    )
    return summary


def score_run(report: dict) -> dict:
    """Use each run's saved expected values; never guess scores for legacy prose."""
    if report.get("protocol") not in (None, SCORING_VERSION):
        raise ValueError(f"Unsupported quality protocol: {report['protocol']}")
    structured = report.get("protocol") == SCORING_VERSION
    evaluations = []
    for result in report["results"]:
        for evaluation in result["evaluations"]:
            item = {"case": result["case"], "probe": evaluation["probe"]}
            if structured:
                expected = evaluation["expected"]
                if not isinstance(expected, dict) or not expected or any(
                    type(value) not in (str, int, bool, type(None))
                    for value in expected.values()
                ):
                    raise ValueError("Expected values must be a nonempty scalar object")
                item["expected"] = expected.copy()
            for mode in ("full", "compacted"):
                item[mode] = (
                    score_answer(evaluation[mode]["answer"], expected)
                    if structured else
                    {"status": "review", "reason": "legacy_freeform_answer"}
                )
            evaluations.append(item)
    if not evaluations:
        raise ValueError("Run contains no evaluations")
    return {
        "scoring_version": SCORING_VERSION,
        "model": report["model"],
        **summarize_evaluations(evaluations),
        "evaluations": evaluations,
    }


def rescore_file(source: Path) -> tuple[Path, dict]:
    original = source.read_bytes()
    scores = score_run(json.loads(original))
    scores["source"] = str(source)
    scores["source_sha256"] = hashlib.sha256(original).hexdigest()
    destination = source.with_suffix(".scores.json")
    with destination.open("x", encoding="utf-8") as output:
        output.write(json.dumps(scores, indent=2, ensure_ascii=False) + "\n")
    return destination, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    for source in args.inputs:
        try:
            destination, scores = rescore_file(source)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"{source}: {error}\n")
        summary = {key: scores[key] for key in ("full", "compacted", "confirmed_regressions")}
        print(f"{destination}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()

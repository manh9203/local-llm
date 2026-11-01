"""Score structured quality answers, or rescore saved runs without MLX.

Usage: python -m benchmarks.scoring RUN.json [RUN.json ...]
Each input gets a small sibling .scores.json report; inputs are never rewritten.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median

from benchmarks.code_grading import DEFAULT_IMAGE, DockerGrader
from benchmarks.quality_cases import case_set_hash, validate_grader

LEGACY_PROTOCOL = "structured-v1"
PROTOCOL_VERSION = "quality-v2"
SCORING_VERSION = "quality-scoring-v3"
STATUSES = ("passed", "failed", "review", "incomplete",
            "configuration_error", "inference_error", "pending")


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


def score_legacy_run(report: dict) -> dict:
    """Use each run's saved expected values; never guess scores for legacy prose."""
    if report.get("protocol") not in (None, LEGACY_PROTOCOL):
        raise ValueError(f"Unsupported quality protocol: {report['protocol']}")
    structured = report.get("protocol") == LEGACY_PROTOCOL
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
                    raise ValueError(
                        "Expected values must be a nonempty scalar object")
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


def grade_answer(response: dict, grader: dict, definition: dict | None = None, code_grader=None) -> dict:
    status = response["status"]
    if status != "completed":
        if status not in STATUSES[3:]:
            raise ValueError(f"Unknown generation status: {status}")
        return {"status": status, "reason": response.get("error", {}).get("message", status)}
    # Never award a clean pass to an output that exhausted its generation budget.
    if response.get("finish_reason") == "length":
        return {"status": "incomplete", "reason": "generation_limit"}
    validate_grader(grader)
    if grader["kind"] == "manual":
        if code_grader is not None:
            return code_grader.grade(definition or {}, response["answer"])
        return {"status": "review", "reason": "manual_review_required"}
    if grader["kind"] == "json":
        score = score_answer(response["answer"], grader["expected"])
        if score["status"] == "failed":
            score["dimension"] = "content" if score["reason"] == "wrong_values" else "format"
        return score
    text = response["answer"].strip()
    lines = text.splitlines()
    violations = []
    if len(lines) != len(grader["line_prefixes"]) or any(
        not line.startswith(prefix) or not line[len(prefix):].strip()
        for line, prefix in zip(lines, grader["line_prefixes"])
    ):
        violations.append("line_structure")
    if len(text.split()) > grader["max_words"]:
        violations.append("word_limit")
    if any(re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE)
           for word in grader["forbidden_words"]):
        violations.append("forbidden_word")
    return ({"status": "failed", "reason": "constraint_violation", "dimension": "format", "constraints": violations}
            if violations else {"status": "passed"})


def summarize_records(records: list[dict], modes: tuple) -> dict:
    summary = {"total_probes": len(records)}
    for mode in modes:
        outcomes = [record[mode] for record in records]
        summary[mode] = {status: sum(
            item["status"] == status for item in outcomes) for status in STATUSES}
        summary[mode].update({
            "format_failures": sum(item.get("dimension") == "format" for item in outcomes),
            "content_failures": sum(item.get("dimension") == "content" for item in outcomes),
        })
    if "compacted" in modes:
        summary["confirmed_regressions"] = sum(
            record["full"]["status"] == "passed" and record["compacted"]["status"] == "failed"
            for record in records
        )
    return summary


def timing_summary(responses: list[dict]) -> dict:
    timings = [response["ttft_seconds"]
               for response in responses if response.get("ttft_seconds") is not None]
    return {
        "attempted_generations": sum(response["status"] != "pending" for response in responses),
        "prompt_tokens": sum(response.get("prompt_tokens") or 0 for response in responses),
        "generated_tokens": sum(response.get("generated_tokens") or 0 for response in responses),
        "generation_seconds": sum(response.get("elapsed_seconds") or 0 for response in responses),
        "median_ttft_seconds": median(timings) if timings else None,
    }


def score_run(report: dict, code_grader=None) -> dict:
    if code_grader is not None and (report.get("protocol") != PROTOCOL_VERSION or report.get("suite") != "capability"):
        raise ValueError(
            "Executable grading requires a quality-v2 capability run")
    if report.get("protocol") != PROTOCOL_VERSION:
        return score_legacy_run(report)
    suite = report["suite"]
    if suite not in ("compaction", "capability"):
        raise ValueError(f"Unknown suite: {suite}")
    definitions = [row["definition"] for row in report["results"]]
    if case_set_hash(suite, definitions) != report["case_set_hash"]:
        raise ValueError("Case-set hash does not match saved definitions")
    modes = ("direct",) if suite == "capability" else ("full", "compacted")
    records = []
    responses = []
    for row in report["results"]:
        definition = row["definition"]
        probes = ([{"name": definition["id"], "grader": definition["grader"]}]
                  if suite == "capability" else [
                      {"name": probe["name"], "grader": {
                          "kind": "json", "expected": probe["expected"]}}
                      for probe in definition["probes"]
        ])
        if [item["probe"] for item in row["evaluations"]] != [probe["name"] for probe in probes]:
            raise ValueError(
                "Saved evaluations do not match the case definitions")
        for probe, evaluation in zip(probes, row["evaluations"], strict=True):
            record = {
                "case": definition["id"], "probe": probe["name"], "category": definition["category"]}
            for mode in modes:
                record[mode] = grade_answer(
                    evaluation[mode], probe["grader"], definition, code_grader)
                responses.append(evaluation[mode])
            records.append(record)
    if not records:
        raise ValueError("Run contains no evaluations")
    scores = {
        "scoring_version": SCORING_VERSION, "protocol": PROTOCOL_VERSION,
        "suite": suite, "model": report["model"], "case_set_hash": report["case_set_hash"],
        "run_status": report["run_status"], "summary": summarize_records(records, modes),
        "categories": {
            category: summarize_records(
                [record for record in records if record["category"] == category], modes)
            for category in sorted({record["category"] for record in records})
        },
        "timing": timing_summary(responses), "evaluations": records,
    }
    if code_grader is not None:
        scores["execution_environment"] = code_grader.environment or {
            "backend": "unavailable", "reason": code_grader.unavailable_reason,
        }
    return scores


def compare_runs(left: dict, right: dict, code_grader=None) -> dict:
    if left.get("protocol") != PROTOCOL_VERSION or right.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(
            "Comparison requires quality-v2 runs with reproducibility metadata")
    # Keep historical experiment settings comparable when reading saved runs.
    for field in ("suite", "case_set_hash", "runtime", "scoring_version", "summary_tokens", "recent_turns_to_keep",
                  "source_excerpts_max_tokens", "source_excerpts_policy"):
        if left.get(field) != right.get(field):
            raise ValueError(f"Incompatible runs: {field} differs")
    left_config = {key: value for key,
                   value in left["config"].items() if key != "model_id"}
    right_config = {key: value for key,
                    value in right["config"].items() if key != "model_id"}
    if left_config != right_config:
        raise ValueError("Incompatible runs: generation configuration differs")
    scores = [score_run(run, code_grader) for run in (left, right)]
    comparison = {
        "scoring_version": SCORING_VERSION, "suite": left["suite"], "case_set_hash": left["case_set_hash"],
        "runs": [{key: score[key] for key in ("model", "run_status", "summary", "categories", "timing")}
                 for score in scores],
        "case_outcomes": [
            {"case": a["case"], "probe": a["probe"], "category": a["category"],
             "left": {mode: a[mode]["status"] for mode in ("direct", "full", "compacted") if mode in a},
             "right": {mode: b[mode]["status"] for mode in ("direct", "full", "compacted") if mode in b}}
            for a, b in zip(scores[0]["evaluations"], scores[1]["evaluations"], strict=True)
        ],
    }
    if code_grader is not None:
        comparison["execution_environment"] = scores[0]["execution_environment"]
    return comparison


def rescore_file(source: Path, output: Path | None = None, code_grader=None) -> tuple[Path, dict]:
    destination = output or source.with_suffix(".scores.json")
    if destination.exists():
        raise FileExistsError(destination)
    original = source.read_bytes()
    scores = score_run(json.loads(original), code_grader)
    scores["source"] = str(source)
    scores["source_sha256"] = hashlib.sha256(original).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        file.write(json.dumps(scores, indent=2, ensure_ascii=False) + "\n")
    return destination, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--compare", type=Path, nargs=2,
                        metavar=("LEFT", "RIGHT"))
    parser.add_argument("--output", type=Path,
                        help="Fresh output path for one score report or a comparison")
    parser.add_argument("--execute-code", action="store_true",
                        help="Opt in to restricted Docker grading")
    parser.add_argument("--code-image", default=DEFAULT_IMAGE,
                        help="Local grader image; never pulled automatically")
    args = parser.parse_args()
    code_grader = DockerGrader(args.code_image) if args.execute_code else None
    if args.compare:
        if args.inputs:
            parser.error("Use either --compare or input files")
        try:
            inputs = [source.read_bytes() for source in args.compare]
            comparison = compare_runs(*(json.loads(data)
                                      for data in inputs), code_grader=code_grader)
            comparison["sources"] = [
                {"path": str(source), "sha256": hashlib.sha256(
                    data).hexdigest()}
                for source, data in zip(args.compare, inputs, strict=True)
            ]
            text = json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as output:
                    output.write(text)
                print(f"Comparison: {args.output}")
            else:
                print(text, end="")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"{error}\n")
        return
    if not args.inputs or (args.output and len(args.inputs) != 1):
        parser.error(
            "Provide input runs; --output requires one input or --compare")
    for source in args.inputs:
        try:
            destination, scores = rescore_file(
                source, args.output, code_grader)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"{source}: {error}\n")
        summary = scores.get("summary") or {
            key: scores[key] for key in ("full", "compacted", "confirmed_regressions")
        }
        print(f"{destination}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()

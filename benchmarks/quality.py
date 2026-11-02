"""Run compaction or capability evaluations with incremental result preservation."""

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from time import perf_counter

from benchmarks.quality_cases import case_manifest, case_set_hash, select_cases
from benchmarks.scoring import PROTOCOL_VERSION, SCORING_VERSION, score_run
from local_llm.config import DEFAULT_MODEL_ID, ChatConfig
from local_llm.conversation.manager import ConversationManager, build_effective_messages
from local_llm.models.registry import get_model_spec

# CLI dependencies stay at module scope. Importing the runner for unit tests
# requires no MLX/Metal initialization; tests supply their own factories.
if __name__ == "__main__":
    from local_llm.conversation.summarizer import MLXSummarizer
    from local_llm.inference.engine import InferenceEngine

RECENT_TURNS_TO_KEEP = 1


def runtime_versions() -> dict:
    packages = {}
    for name in ("mlx", "mlx-lm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": python_version(), **packages}


def model_metadata(engine) -> dict:
    tokenizer = getattr(engine, "tokenizer", None)
    name = str(getattr(tokenizer, "name_or_path", ""))
    parts = Path(name).parts
    revision = None
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        revision = parts[parts.index("snapshots") + 1]
    if revision is None:
        cache = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                str(
                    Path(
                        os.environ.get(
                            "HF_HOME", str(Path.home() / ".cache/huggingface")
                        )
                    )
                    / "hub"
                ),
            )
        )
        ref = (
            cache
            / ("models--" + engine.config.model_id.replace("/", "--"))
            / "refs/main"
        )
        try:
            if ref.is_file():
                revision = ref.read_text().strip()
        except OSError:
            revision = None
    template = getattr(tokenizer, "chat_template", None)
    return {
        "revision": revision,
        "revision_source": "tokenizer_path"
        if "snapshots" in parts
        else "cache_ref"
        if revision
        else None,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()
        if isinstance(template, str)
        else None,
    }


def is_fatal(error: Exception) -> bool:
    return isinstance(error, MemoryError) or any(
        text in str(error).casefold()
        for text in (
            "out of memory",
            "failed to allocate",
            "allocation failed",
            "device lost",
            "no metal device",
        )
    )


def failure(error: Exception, stage: str, **partial) -> dict:
    return {
        **partial,
        "status": "configuration_error"
        if stage == "prepare" and isinstance(error, ValueError)
        else "inference_error",
        "error": {"type": type(error).__name__, "message": str(error), "stage": stage},
        "fatal_error": is_fatal(error),
    }


def generate_answer(engine, messages: list[dict], max_new_tokens: int = 128) -> dict:
    session = None
    chunks = []
    result = {
        "answer": "",
        "prompt_tokens": None,
        "generated_tokens": 0,
        "ttft_seconds": None,
        "elapsed_seconds": None,
        "finish_reason": None,
    }
    stage = "prepare"
    started = None
    try:
        session = engine.create_session()
        session.config = replace(engine.config, max_new_tokens=max_new_tokens)
        prepared = session.prepare_messages(messages)
        result["prompt_tokens"] = getattr(
            prepared.context_stats, "original_tokens", len(prepared.token_ids)
        )
        if prepared.context_stats.truncated or prepared.messages != messages:
            raise ValueError(
                "Evaluation prompt exceeds its input budget; task was not run"
            )
        stage = "generate"
        started = perf_counter()
        # extend retains yielded chunks even when the stream raises midway.
        chunks.extend(session.stream_prepared(prepared))
        if session.last_stats is None:
            raise RuntimeError("Generation returned no statistics")
        stats = session.last_stats
        result.update(
            generated_tokens=stats["generation_tokens"],
            ttft_seconds=stats.get("first_token_seconds"),
            elapsed_seconds=perf_counter() - started,
            finish_reason=stats["finish_reason"],
            peak_memory_gb=stats.get("peak_memory_gb"),
        )
        result["status"] = (
            "incomplete" if stats["finish_reason"] == "length" else "completed"
        )
    except Exception as error:  # noqa: BLE001 - Preserve partial output and record model failures.
        if started is not None:
            result["elapsed_seconds"] = perf_counter() - started
        if session is not None:
            result["generated_tokens"] = len(
                getattr(session, "last_generated_tokens", [])
            )
        result = failure(error, stage, **result)
    finally:
        result["answer"] = "".join(chunks).strip()
        if session is not None:
            session.clear_cache()
    return result


def with_system(config: ChatConfig, messages: list[dict]) -> list[dict]:
    copied = [message.copy() for message in messages]
    if copied[0]["role"] != "system":
        copied.insert(0, {"role": "system", "content": config.system_prompt})
    return copied


def run_compaction_case(engine, row: dict, summary_tokens: int, summarizer_factory):
    definition = row["definition"]
    manager = ConversationManager()
    history = [{"role": "system", "content": engine.config.system_prompt}]
    previous_summary = None
    previous_covered = 1
    try:
        summarizer = summarizer_factory(engine)
        for number, segment in enumerate(definition["segments"], start=1):
            history.extend(message.copy() for message in segment)
            completed_turns = (len(history) - 1) // 2
            covered_count = 1 + 2 * (completed_turns - RECENT_TURNS_TO_KEEP)
            started = perf_counter()
            summary = summarizer.summarize(
                messages=history[previous_covered:covered_count],
                previous_summary=previous_summary,
                max_tokens=summary_tokens,
            )
            checkpoint = manager.create(
                messages=history,
                covered_count=covered_count,
                summary=summary,
                model_id=engine.config.model_id,
            )
            row["rounds"].append(
                {
                    "round": number,
                    "covered_count": covered_count,
                    "summary": summary,
                    "summary_seconds": perf_counter() - started,
                    "stats": asdict(summarizer.last_stats)
                    if summarizer.last_stats
                    else None,
                }
            )
            previous_summary, previous_covered = summary, covered_count
            print(f"  Round {number}: covered={covered_count}", flush=True)
        for probe, evaluation in zip(
            definition["probes"], row["evaluations"], strict=True
        ):
            question = (
                probe["question"]
                + definition["instructions"]
                + json.dumps(probe["answer_format"])
            )
            evaluation["question"] = question
            full = history + [{"role": "user", "content": question}]
            compacted = build_effective_messages(
                full, checkpoint, engine.config.model_id
            )
            for mode, messages in (("full", full), ("compacted", compacted)):
                evaluation[mode] = generate_answer(
                    engine, messages, definition["max_new_tokens"]
                )
                if evaluation[mode].get("fatal_error"):
                    return
            print(f"  {probe['name']}: evaluated full and compacted", flush=True)
    except Exception as error:  # noqa: BLE001 - Record a failed summary without losing other cases.
        for evaluation in row["evaluations"]:
            for mode in ("full", "compacted"):
                if evaluation[mode]["status"] == "pending":
                    evaluation[mode] = failure(error, "compaction", answer="")


def save_progress(path: Path, report: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_benchmark(
    config: ChatConfig,
    suite: str,
    cases: tuple,
    output: Path,
    engine_factory,
    summarizer_factory=None,
    summary_tokens: int = 512,
    max_probes: int | None = None,
) -> dict:
    definitions = case_manifest(suite, cases, max_probes)
    if config.use_cache or config.enable_thinking or config.temperature != 0:
        raise ValueError(
            "Quality evaluation requires cache/thinking off and temperature zero"
        )
    rows = []
    for definition in definitions:
        modes = ("direct",) if suite == "capability" else ("full", "compacted")
        probes = (
            [{"name": definition["id"]}]
            if suite == "capability"
            else definition["probes"]
        )
        rows.append(
            {
                "definition": definition,
                "rounds": [],
                "evaluations": [
                    {
                        "probe": probe["name"],
                        **{mode: {"status": "pending"} for mode in modes},
                    }
                    for probe in probes
                ],
            }
        )
    report = {
        "protocol": PROTOCOL_VERSION,
        "scoring_version": SCORING_VERSION,
        "suite": suite,
        "model": config.model_id,
        "config": asdict(config),
        "case_set_hash": case_set_hash(suite, definitions),
        "runtime": runtime_versions(),
        "run_status": "running",
        "results": rows,
    }
    if suite == "compaction":
        report.update(
            summary_tokens=summary_tokens, recent_turns_to_keep=RECENT_TURNS_TO_KEEP
        )
        if summarizer_factory is None:
            raise ValueError("Compaction requires a summarizer factory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    try:
        engine = engine_factory(config)
        report["model_metadata"] = model_metadata(engine)
        save_progress(output, report)
        for row in rows:
            definition = row["definition"]
            print(f"\n=== {definition['id']} ===", flush=True)
            if suite == "capability":
                row["evaluations"][0]["direct"] = generate_answer(
                    engine,
                    with_system(config, definition["messages"]),
                    definition["max_new_tokens"],
                )
            else:
                run_compaction_case(engine, row, summary_tokens, summarizer_factory)
            save_progress(output, report)
            if any(
                response.get("fatal_error")
                for evaluation in row["evaluations"]
                for response in evaluation.values()
                if isinstance(response, dict)
            ):
                report["run_status"] = "aborted"
                break
        else:
            report["run_status"] = "completed"
    except BaseException as error:
        report["run_status"] = "aborted"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save_progress(output, report)
        if not isinstance(error, Exception):
            raise
    save_progress(output, report)
    return report


def main(engine_factory, summarizer_factory):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("compaction", "capability"), default="compaction"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--case", default="all")
    parser.add_argument("--category", default="all")
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--summary-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        get_model_spec(args.model)
        if args.max_probes is not None and (
            args.max_probes < 1 or args.suite != "compaction"
        ):
            raise ValueError("--max-probes is a positive compaction-only option")
        if not 0 < args.summary_tokens < 8192:
            raise ValueError("--summary-tokens must be between 1 and 8191")
        cases = select_cases(args.suite, args.case, args.category)
        # Validate before loading weights or creating a result file.
        case_manifest(args.suite, cases, args.max_probes)
        config = ChatConfig(
            model_id=args.model,
            temperature=0,
            enable_thinking=False,
            use_cache=False,
            max_context_tokens=8192,
            max_new_tokens=128,
        )
        report = run_benchmark(
            config,
            args.suite,
            cases,
            args.output,
            engine_factory,
            summarizer_factory,
            args.summary_tokens,
            args.max_probes,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    scores = score_run(report)
    print(
        json.dumps(
            {key: scores[key] for key in ("suite", "run_status", "summary")}, indent=2
        )
    )
    print(f"Results: {args.output}")
    if report["run_status"] == "aborted":
        parser.exit(1, "Evaluation aborted; completed and pending cases are saved.\n")


if __name__ == "__main__":
    main(InferenceEngine, MLXSummarizer)

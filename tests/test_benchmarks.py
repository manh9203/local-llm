import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import code_grading
from benchmarks.code_grading import DockerGrader, container_command, extract_code
from benchmarks.quality import generate_answer, run_benchmark
from benchmarks.quality_cases import (
    ALL_CASES,
    CAPABILITY_CASES,
    case_manifest,
    case_set_hash,
    select_cases,
)
from benchmarks.scoring import (
    LEGACY_PROTOCOL,
    compare_runs,
    grade_answer,
    rescore_file,
    score_answer,
    score_run,
)
from local_llm.config import ChatConfig

REFERENCE_PYTHON = """from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity")
        self.capacity, self.data = capacity, OrderedDict()
    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]
    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
"""

REFERENCE_CPP = """#include <cstddef>
#include <memory>
#include <utility>
class Buffer {
    std::unique_ptr<int[]> data_;
    std::size_t size_;
public:
    explicit Buffer(std::size_t n): data_(std::make_unique<int[]>(n)), size_(n) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& b) noexcept: data_(std::move(b.data_)), size_(std::exchange(b.size_, 0)) {}
    Buffer& operator=(Buffer&& b) noexcept {
        if (this != &b) { data_ = std::move(b.data_); size_ = std::exchange(b.size_, 0); }
        return *this;
    }
    std::size_t size() const { return size_; }
    int& operator[](std::size_t i) { return data_[i]; }
    const int& operator[](std::size_t i) const { return data_[i]; }
};
"""


# Independent correct/incorrect examples cover the actual benchmark decisions.
# In particular, mentioning a fact in a denial must not earn credit for it.
@pytest.mark.parametrize(
    "case_name,probe_name,correct,incorrect",
    [
        (
            "technical",
            "current_context",
            '{"context_tokens":8192,"generation_tokens":512}',
            '{"context_tokens":4096,"generation_tokens":256}',
        ),
        (
            "technical",
            "checkpoint_design",
            '{"algorithm":"sha256","input":"message_prefix"}',
            '{"algorithm":"sha256","input":"conversation_id"}',
        ),
        ("technical", "recent_turns", '{"completed_turns":4}', '{"completed_turns":1}'),
        (
            "travel",
            "destination",
            '{"destination":"ottawa"}',
            '{"destination":"kingston"}',
        ),
        (
            "travel",
            "transport_budget",
            '{"transport":"train","budget_cad":500}',
            '{"transport":"train","budget_cad":400}',
        ),
        ("travel", "museum_booking", '{"booked":false}', '{"booked":true}'),
        (
            "travel",
            "museum_decision",
            '{"decision":"undecided"}',
            '{"decision":"do_not_book"}',
        ),
        (
            "travel",
            "hotel_name",
            '{"hotel_name":null}',
            '{"hotel_name":"Downtown Inn"}',
        ),
        ("project", "demo_deadline", '{"day":"wednesday"}', '{"day":"monday"}'),
        (
            "project",
            "verification",
            '{"approved":"university_email","deferred":"government_id"}',
            '{"approved":"government_id","deferred":"university_email"}',
        ),
        (
            "project",
            "next_priority",
            '{"priority":"verification_testing"}',
            '{"priority":"reviews"}',
        ),
        ("project", "mobile_scope", '{"in_scope":false}', '{"in_scope":true}'),
        (
            "project",
            "completion_status",
            '{"verification":"not_confirmed","listing_filters":"completed"}',
            '{"verification":"completed","listing_filters":"completed"}',
        ),
        (
            "project",
            "completion_status",
            '{"verification":"not_confirmed","listing_filters":"completed"}',
            '{"verification":"not_confirmed","listing_filters":"not_confirmed"}',
        ),
    ],
)
def test_quality_decisions(case_name, probe_name, correct, incorrect):
    case = next(case for case in ALL_CASES if case.name == case_name)
    probe = next(probe for probe in case.probes if probe.name == probe_name)
    assert score_answer(correct, probe.expected)["status"] == "passed"
    assert score_answer(incorrect, probe.expected)["status"] == "failed"


@pytest.mark.parametrize(
    "answer",
    [
        "The context is 4096, not 8192 tokens.",
        '{"context_tokens":8192',
        '{"context_tokens":4096,"context_tokens":8192}',
        '{"context_tokens":"8192"}',
        '{"context_tokens":8192,"explanation":"actually 4096"}',
        '[{"context_tokens":8192}]',
        "{}",
    ],
)
def test_noncompliant_or_contradictory_answers_fail(answer):
    assert score_answer(answer, {"context_tokens": 8192})["status"] == "failed"


def test_json_types_are_not_coerced_and_single_code_fence_is_accepted():
    assert score_answer('{"booked":0}', {"booked": False})["reason"] == "wrong_types"
    assert score_answer('{"turns":true}', {"turns": 1})["status"] == "failed"
    assert score_answer('```json\n{"booked":false}\n```', {"booked": False}) == {
        "status": "passed"
    }


def test_string_values_ignore_case_but_field_names_remain_exact():
    assert score_answer('{"destination":" Ottawa "}', {"destination": "ottawa"}) == {
        "status": "passed"
    }
    assert (
        score_answer('{"Destination":"Ottawa"}', {"destination": "ottawa"})["status"]
        == "failed"
    )


@pytest.fixture
def raw_run():
    return {
        "protocol": LEGACY_PROTOCOL,
        "model": "test-model",
        "config": {"temperature": 0},
        "results": [
            {
                "case": "project",
                "history": [{"role": "user", "content": "Original history"}],
                "rounds": [{"summary": "Original summary", "summary_seconds": 1.5}],
                "evaluations": [
                    {
                        "probe": "verification",
                        "expected": {
                            "approved": "university_email",
                            "deferred": "government_id",
                        },
                        "full": {
                            "answer": '{"approved":"university_email","deferred":"government_id"}',
                            "prompt_tokens": 123,
                            "elapsed_seconds": 0.5,
                        },
                        "compacted": {
                            "answer": '{"approved":"government_id","deferred":"university_email"}',
                            "prompt_tokens": 80,
                            "elapsed_seconds": 0.3,
                        },
                    }
                ],
            }
        ],
    }


def test_scoring_preserves_raw_data_and_counts_confirmed_regressions(raw_run):
    original = deepcopy(raw_run)
    scores = score_run(raw_run)
    assert raw_run == original
    assert scores["total_probes"] == 1
    assert scores["full"] == {"passed": 1, "failed": 0, "review": 0}
    assert scores["compacted"] == {"passed": 0, "failed": 1, "review": 0}
    assert scores["confirmed_regressions"] == 1
    assert "results" not in scores
    assert "answer" not in scores["evaluations"][0]["full"]


def test_legacy_prose_requires_review_even_when_old_scorer_said_pass(raw_run):
    raw_run.pop("protocol")
    evaluation = raw_run["results"][0]["evaluations"][0]
    evaluation.pop("expected")
    evaluation["full"] = {
        "answer": "Government-ID approved; university-email deferred.",
        "passed": True,
    }
    evaluation["compacted"] = {
        "answer": "No, museum tickets have not been booked.",
        "passed": False,
    }
    scores = score_run(raw_run)
    assert (
        scores["full"] == scores["compacted"] == {"passed": 0, "failed": 0, "review": 1}
    )
    assert scores["confirmed_regressions"] == 0


def test_unknown_protocol_and_missing_truth_are_rejected(raw_run):
    raw_run["protocol"] = "future-version"
    with pytest.raises(ValueError, match="Unsupported"):
        score_run(raw_run)
    raw_run["protocol"] = LEGACY_PROTOCOL
    raw_run["results"][0]["evaluations"][0]["expected"] = {}
    with pytest.raises(ValueError, match="Expected values"):
        score_run(raw_run)


def test_rescore_file_preserves_original_and_existing_outputs(tmp_path, raw_run):
    source = tmp_path / "run.json"
    original = json.dumps(raw_run).encode()
    source.write_bytes(original)
    destination, scores = rescore_file(source)
    assert destination == source.with_suffix(".scores.json")
    assert json.loads(destination.read_text()) == scores
    assert scores["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original
    saved = destination.read_bytes()
    with pytest.raises(FileExistsError):
        rescore_file(source)
    assert source.read_bytes() == original
    assert destination.read_bytes() == saved


def test_rescore_cli_works_without_mlx_or_site_packages(tmp_path, raw_run):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(raw_run))
    command = [sys.executable, "-S", "-m", "benchmarks.scoring", str(source)]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert ".scores.json" in completed.stdout
    assert (
        json.loads(source.with_suffix(".scores.json").read_text())[
            "confirmed_regressions"
        ]
        == 1
    )


class FakeSession:
    def __init__(self, engine, script):
        self.config = engine.config
        self.script = script
        self.last_stats = None
        self.last_generated_tokens = []
        self.cache_cleared = False

    def prepare_messages(self, messages):
        self.prepared_messages = messages
        return SimpleNamespace(
            messages=messages,
            token_ids=list(range(10)),
            context_stats=SimpleNamespace(
                truncated=self.script.get("truncated", False)
            ),
        )

    def stream_prepared(self, prepared):
        for chunk in self.script.get("chunks", [self.script.get("answer", "")]):
            self.last_generated_tokens.append(1)
            yield chunk
        if "error" in self.script:
            raise self.script["error"]
        self.last_stats = {
            "generation_tokens": len(self.last_generated_tokens),
            "first_token_seconds": 0.01,
            "peak_memory_gb": 1.0,
            "finish_reason": self.script.get("finish_reason", "stop"),
        }

    def clear_cache(self):
        self.cache_cleared = True


class FakeEngine:
    def __init__(self, config, scripts):
        self.config = config
        self.scripts = iter(scripts)
        self.sessions = []

    def create_session(self):
        session = FakeSession(self, next(self.scripts))
        self.sessions.append(session)
        return session


def evaluation_config():
    return ChatConfig(
        model_id="test-model", temperature=0, use_cache=False, enable_thinking=False
    )


@pytest.mark.parametrize(
    "change",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 8192},
        {"category": "missing"},
        {"messages": ()},
        {"messages": ({"role": "assistant", "content": "Wrong role"},)},
        {"grader": {"kind": "executable"}},
        {"grader": {"kind": "json", "expected": {}}},
        {"grader": {"kind": "manual", "rubric": [], "reference": ""}},
    ],
)
def test_invalid_capability_cases_are_rejected_with_case_id(change):
    case = replace(CAPABILITY_CASES[0], **change)
    with pytest.raises(ValueError, match=case.id):
        case_manifest("capability", (case,))


def test_case_selection_uniqueness_and_content_hash():
    assert {case.id for case in select_cases("capability", category="python")} == {
        "python_lru_cache",
        "python_ttl_cache",
    }
    assert len(CAPABILITY_CASES) == 10
    assert all(
        len(select_cases("capability", category=category)) == 2
        for category in ("recall", "instructions", "python", "cpp", "systems")
    )
    assert select_cases("compaction", case_id="technical") == (ALL_CASES[0],)
    with pytest.raises(ValueError, match="No capability cases"):
        select_cases("capability", case_id="missing")
    with pytest.raises(ValueError, match="unique"):
        case_manifest("capability", (CAPABILITY_CASES[0],) * 2)
    original = case_manifest("capability", CAPABILITY_CASES)
    assert case_set_hash("capability", original) == case_set_hash(
        "capability", json.loads(json.dumps(original))
    )
    for field, replacement in (
        ("messages", [{"role": "user", "content": "Changed task"}]),
        ("max_new_tokens", 512),
        ("grader", {"kind": "manual", "rubric": ["Changed"]}),
    ):
        changed = deepcopy(original)
        changed[0][field] = replacement
        assert case_set_hash("capability", changed) != case_set_hash(
            "capability", original
        )


@pytest.mark.parametrize(
    "answer,passed",
    [
        ("Action: Remove the expired entry.\nCheck: Confirm it is absent.", True),
        ("Action: Remove it the fastest way.\nCheck: Verify deletion.", True),
        ("Action: Remove it fast.\nCheck: Verify deletion.", False),
        ("Action: Remove it.\n\nCheck: Verify deletion.", False),
        ("Action: Remove it.\nCheck: Verify deletion.\nExtra line.", False),
        ("Action: \nCheck: Verify deletion.", False),
        ("Action: " + "word " * 33 + "\nCheck: Verify deletion.", False),
    ],
)
def test_instruction_constraints_are_explicit_not_semantic_keywords(answer, passed):
    score = grade_answer(
        {"status": "completed", "answer": answer, "finish_reason": "stop"},
        CAPABILITY_CASES[1].grader,
    )
    assert (score["status"] == "passed") is passed


def test_numeric_reference_includes_rounding_and_insufficient_information():
    grader = select_cases("capability", case_id="systems_memory_budget")[0].grader
    assert (
        grade_answer(
            {
                "status": "completed",
                "answer": '{"max_tokens_per_session":13653,"safe_inflight_requests":null}',
            },
            grader,
        )["status"]
        == "passed"
    )
    assert (
        grade_answer(
            {
                "status": "completed",
                "answer": '{"max_tokens_per_session":13654,"safe_inflight_requests":3}',
            },
            grader,
        )["status"]
        == "failed"
    )


def test_runner_preserves_partial_answers_continues_and_applies_case_budgets(tmp_path):
    config = evaluation_config()
    engine = FakeEngine(
        config,
        [
            {
                "answer": '{"context_tokens":8192,"generation_tokens":512,"algorithm":"sha256"}'
            },
            {
                "chunks": ["partial answer"],
                "error": RuntimeError("Recoverable failure"),
            },
            {"answer": "raise RuntimeError('must not be executed')"},
            {"answer": "C++ code requiring manual review"},
            {
                "answer": '{"max_tokens_per_session":13653,"safe_inflight_requests":null}',
                "finish_reason": "length",
            },
        ],
    )
    output = tmp_path / "run.json"
    report = run_benchmark(
        config, "capability", CAPABILITY_CASES[:5], output, lambda _: engine
    )
    assert report == json.loads(output.read_text())
    assert report["run_status"] == "completed"
    scores = score_run(report)
    summary = scores["summary"]["direct"]
    assert summary["passed"] == 1
    assert summary["review"] == 2
    assert summary["inference_error"] == 1
    assert summary["incomplete"] == 1
    partial = report["results"][1]["evaluations"][0]["direct"]
    assert partial["answer"] == "partial answer"
    assert partial["generated_tokens"] == 1
    assert partial["error"]["stage"] == "generate"
    assert [session.config.max_new_tokens for session in engine.sessions] == [
        256,
        256,
        1024,
        1024,
        256,
    ]
    assert all(session.cache_cleared for session in engine.sessions)
    assert engine.sessions[2].prepared_messages[1:] == list(
        CAPABILITY_CASES[2].messages
    )
    assert config.max_new_tokens == 512
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(config, "capability", CAPABILITY_CASES, output, lambda _: engine)
    assert output.read_bytes() == saved


def test_truncated_prompts_are_not_generated_or_graded():
    engine = FakeEngine(evaluation_config(), [{"truncated": True}])
    response = generate_answer(engine, [{"role": "user", "content": "Too long"}], 256)
    assert response["status"] == "configuration_error"
    assert response["answer"] == ""
    assert engine.sessions[0].last_generated_tokens == []
    assert engine.sessions[0].cache_cleared


@pytest.mark.parametrize(
    "error", [MemoryError("OOM"), RuntimeError("Metal allocation failed")]
)
def test_fatal_generation_preserves_partial_progress_and_pending_cases(tmp_path, error):
    engine = FakeEngine(evaluation_config(), [{"chunks": ["partial"], "error": error}])
    output = tmp_path / "aborted.json"
    report = run_benchmark(
        engine.config, "capability", CAPABILITY_CASES, output, lambda _: engine
    )
    assert report["run_status"] == "aborted"
    assert report["results"][0]["evaluations"][0]["direct"]["answer"] == "partial"
    scores = score_run(json.loads(output.read_text()))
    assert scores["summary"]["direct"]["inference_error"] == 1
    assert scores["summary"]["direct"]["pending"] == len(CAPABILITY_CASES) - 1


def test_model_load_error_preserves_scheduled_cases(tmp_path):
    def broken_factory(config):
        raise RuntimeError("Model cannot load")

    output = tmp_path / "load-error.json"
    report = run_benchmark(
        evaluation_config(), "capability", CAPABILITY_CASES, output, broken_factory
    )
    assert report["run_status"] == "aborted"
    assert score_run(json.loads(output.read_text()))["summary"]["direct"][
        "pending"
    ] == len(CAPABILITY_CASES)
    assert report["error"]["message"] == "Model cannot load"


@pytest.fixture
def capability_run(tmp_path):
    config = evaluation_config()
    engine = FakeEngine(
        config,
        [
            {
                "answer": '{"context_tokens":8192,"generation_tokens":512,"algorithm":"sha256"}'
            }
        ],
    )
    return run_benchmark(
        config,
        "capability",
        CAPABILITY_CASES[:1],
        tmp_path / "compare.json",
        lambda _: engine,
    )


def test_compare_reports_same_tasks_without_mixing_model_metadata(capability_run):
    other = deepcopy(capability_run)
    other["model"] = other["config"]["model_id"] = "other-model"
    other["model_metadata"]["revision"] = "other-revision"
    other["results"][0]["evaluations"][0]["direct"]["answer"] = (
        '{"context_tokens":4096,"generation_tokens":512,"algorithm":"sha256"}'
    )
    comparison = compare_runs(capability_run, other)
    assert comparison["runs"][0]["summary"]["direct"]["passed"] == 1
    assert comparison["runs"][1]["summary"]["direct"]["content_failures"] == 1
    assert comparison["case_outcomes"][0]["right"] == {"direct": "failed"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("suite", "compaction"),
        ("case_set_hash", "different"),
        ("runtime", {"mlx": "different"}),
        ("scoring_version", "different"),
        ("source_excerpts_max_tokens", 384),
        ("source_excerpts_policy", "different"),
    ],
)
def test_compare_rejects_incompatible_runs(capability_run, field, value):
    other = deepcopy(capability_run)
    other[field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        compare_runs(capability_run, other)


def test_compare_rejects_different_settings_and_modified_truth(capability_run):
    other = deepcopy(capability_run)
    other["config"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="configuration"):
        compare_runs(capability_run, other)
    other = deepcopy(capability_run)
    other["results"][0]["definition"]["grader"]["expected"]["context_tokens"] = 4096
    with pytest.raises(ValueError, match="hash"):
        score_run(other)


def test_v2_offline_scoring_and_comparison_cli(tmp_path, capability_run):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    for path in (left, right):
        path.write_text(json.dumps(capability_run))
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-S", "-m", "benchmarks.scoring"]
    subprocess.run([*command, str(left)], cwd=root, check=True, capture_output=True)
    assert (
        json.loads(left.with_suffix(".scores.json").read_text())["summary"]["direct"][
            "passed"
        ]
        == 1
    )
    compared = subprocess.run(
        [*command, "--compare", str(left), str(right)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(json.loads(compared.stdout)["runs"]) == 2


@pytest.mark.parametrize(
    "answer",
    [
        "Here is code:\n```python\npass\n```",
        "```python\n\n```",
        "```python\npass\n```\n```python\npass\n```",
        "pass",
    ],
)
def test_code_extraction_rejects_ambiguous_submissions(answer):
    with pytest.raises(ValueError):
        extract_code(answer, "python")


def test_container_command_does_not_expose_host_directories_or_network(tmp_path):
    command = container_command("sha256:fixed", "test", tmp_path, ["/input/run.sh"])
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in command
    assert "no-new-privileges" in command
    assert command[command.index("--pull") + 1] == "never"
    assert command.count("--mount") == 1
    assert (
        command[command.index("--mount") + 1]
        == f"type=bind,src={tmp_path},dst=/input,readonly"
    )
    assert all(
        flag in command
        for flag in ("--memory", "--memory-swap", "--pids-limit", "--cpus", "--ulimit")
    )
    assert "/var/run/docker.sock" not in " ".join(command)
    assert "--privileged" not in command


def test_unavailable_isolation_and_changed_tasks_require_review(monkeypatch):
    monkeypatch.setattr(code_grading.shutil, "which", lambda _: None)
    grader = DockerGrader()
    definition = code_grading.KNOWN_CASES["python_lru_cache"]
    assert grader.grade(definition, "anything")["reason"] == "isolation_unavailable"
    changed = deepcopy(definition)
    changed["messages"][0]["content"] = "A different task"
    assert grader.grade(changed, "anything")["reason"] == "no_matching_executable_tests"


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"returncode": 0, "output": code_grading.SUCCESS}, "passed"),
        ({"returncode": 0, "output": "premature exit without running tests"}, "failed"),
        (
            {
                "returncode": 97,
                "output": "Conditional jump depends on uninitialised value",
            },
            "failed",
        ),
        ({"returncode": 125, "output": "Docker unavailable"}, "review"),
        ({"returncode": None, "timed_out": True, "output": ""}, "failed"),
        ({"returncode": -25, "output_limit": True, "output": ""}, "failed"),
    ],
)
def test_execution_results_do_not_confuse_infrastructure_with_answer_failures(
    monkeypatch, result, expected
):
    grader = DockerGrader.__new__(DockerGrader)
    grader.environment = {"image_id": "sha256:test"}
    grader.unavailable_reason = None
    execution = {"timed_out": False, "output_limit": False, **result}
    monkeypatch.setattr(code_grading, "run_container", lambda *args: execution)
    answer = "```python\n" + REFERENCE_PYTHON + "```"
    outcome = grader.grade(code_grading.KNOWN_CASES["python_lru_cache"], answer)
    assert outcome["status"] == expected
    assert outcome["execution"] == execution


def test_incomplete_outputs_are_not_executed():
    class NeverExecute:
        def grade(self, *args):
            raise AssertionError("incomplete generation must not execute")

    result = grade_answer(
        {"status": "completed", "finish_reason": "length", "answer": "code"},
        CAPABILITY_CASES[2].grader,
        code_grading.KNOWN_CASES["python_lru_cache"],
        NeverExecute(),
    )
    assert result["status"] == "incomplete"


@pytest.fixture
def docker_grader():
    if os.environ.get("RUN_CODE_INTEGRATION") != "1":
        pytest.skip("Docker execution integration disabled")
    grader = DockerGrader()
    assert grader.environment is not None, grader.unavailable_reason
    return grader


@pytest.mark.parametrize(
    "case_id,code,expected",
    [
        ("python_lru_cache", REFERENCE_PYTHON, "passed"),
        (
            "python_lru_cache",
            REFERENCE_PYTHON.replace(
                "self.data.move_to_end(key)\n        return",
                "# missing promotion\n        return",
            ),
            "failed",
        ),
        ("python_lru_cache", "raise SystemExit(0)", "failed"),
        ("cpp_move_only_buffer", REFERENCE_CPP, "passed"),
        (
            "cpp_move_only_buffer",
            REFERENCE_CPP.replace(
                "std::make_unique<int[]>(n)", "std::unique_ptr<int[]>(new int[n])"
            ),
            "failed",
        ),
        (
            "cpp_move_only_buffer",
            REFERENCE_CPP.replace("size_(n)", "size_(0)"),
            "failed",
        ),
    ],
)
def test_docker_reference_and_regression_cases(docker_grader, case_id, code, expected):
    definition = code_grading.KNOWN_CASES[case_id]
    label = "python" if definition["category"] == "python" else "cpp"
    result = docker_grader.grade(definition, f"```{label}\n{code}\n```")
    assert result["status"] == expected, result
    if "new int[n]" in code:
        assert "uninitialised" in result["execution"]["output"], result


def test_docker_filesystem_and_network_boundaries(docker_grader, tmp_path):
    canary = tmp_path / "host-only-canary"
    canary.write_text("host data")
    probe = f"""import os
import pathlib
import socket
assert os.getuid() == 65534
assert not any(os.environ.get(key) for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'))
assert not pathlib.Path({str(canary)!r}).exists()
try:
    pathlib.Path('/input/answer.py').write_text('overwrite')
except OSError:
    pass
else:
    raise AssertionError('input mount is writable')
try:
    connection = socket.create_connection(('1.1.1.1', 53), timeout=0.2)
except OSError:
    pass
else:
    connection.close()
    raise AssertionError('external network is available')
"""
    result = docker_grader.grade(
        code_grading.KNOWN_CASES["python_lru_cache"],
        f"```python\n{probe + REFERENCE_PYTHON}\n```",
    )
    assert result["status"] == "passed", result
    assert canary.read_text() == "host data"


def test_docker_timeout_stops_and_removes_container(docker_grader, monkeypatch):
    monkeypatch.setitem(code_grading.LIMITS, "wall_seconds", 2)
    result = docker_grader.grade(
        code_grading.KNOWN_CASES["python_lru_cache"], "```python\nwhile True: pass\n```"
    )
    assert result["reason"] == "execution_limit", result
    assert result["execution"]["timed_out"]


def test_executable_scoring_is_opt_in_and_records_provenance(capability_run, tmp_path):
    run = deepcopy(capability_run)
    run["results"][0]["definition"] = deepcopy(
        code_grading.KNOWN_CASES["python_lru_cache"]
    )
    evaluation = run["results"][0]["evaluations"][0]
    evaluation["probe"] = "python_lru_cache"
    evaluation["direct"]["answer"] = f"```python\n{REFERENCE_PYTHON}\n```"
    run["case_set_hash"] = case_set_hash(
        "capability", [run["results"][0]["definition"]]
    )
    assert score_run(run)["summary"]["direct"]["review"] == 1

    class CheckedBackend:
        unavailable_reason = None

        def __init__(self):
            self.environment = {
                "backend": "docker",
                "image_id": "sha256:fixed",
                "test_suite_sha256": "tests",
            }

        def grade(self, definition, answer):
            assert definition == code_grading.KNOWN_CASES["python_lru_cache"]
            assert answer.startswith("```python")
            return {"status": "passed", "reason": "code_tests_passed"}

    backend = CheckedBackend()
    source = tmp_path / "raw.json"
    original = json.dumps(run).encode()
    source.write_bytes(original)
    destination, scores = rescore_file(source, tmp_path / "executed.json", backend)
    assert source.read_bytes() == original
    assert scores["summary"]["direct"]["passed"] == 1
    assert scores["execution_environment"] == backend.environment
    assert json.loads(destination.read_text()) == scores
    comparison = compare_runs(run, run, backend)
    assert comparison["execution_environment"] == backend.environment
    assert all(
        result["summary"]["direct"]["review"] == 0 for result in comparison["runs"]
    )


def test_docker_input_mount_works_with_private_host_umask(docker_grader):
    previous = os.umask(0o077)
    try:
        result = docker_grader.grade(
            code_grading.KNOWN_CASES["python_lru_cache"],
            f"```python\n{REFERENCE_PYTHON}\n```",
        )
    finally:
        os.umask(previous)
    assert result["status"] == "passed", result


def test_docker_output_is_bounded_on_the_host(docker_grader):
    code = 'print("x" * 1048576)\n' + REFERENCE_PYTHON
    result = docker_grader.grade(
        code_grading.KNOWN_CASES["python_lru_cache"], f"```python\n{code}\n```"
    )
    assert result["reason"] == "execution_limit", result
    assert result["execution"]["output_limit"]
    assert (
        len(result["execution"]["output"].encode())
        <= code_grading.LIMITS["output_bytes"]
    )


REFERENCE_TTL = """class TTLCache:
    def __init__(self, ttl, clock):
        if ttl <= 0:
            raise ValueError("ttl")
        self.ttl, self.clock, self.data = ttl, clock, {}
    def put(self, key, value):
        self.data[key] = (value, self.clock() + self.ttl)
    def get(self, key):
        if key not in self.data:
            return None
        value, deadline = self.data[key]
        if self.clock() >= deadline:
            del self.data[key]
            return None
        return value
"""

REFERENCE_PARSE = """#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>
bool parse_int32(std::string_view text, std::int32_t& value) noexcept {
    if (text.empty()) return false;
    if (text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || text.front() == '+' || text.front() == '-') return false;
    }
    std::int32_t parsed = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return false;
    value = parsed;
    return true;
}
"""

OVERFLOW_PARSE = """#include <cstdint>
#include <string_view>
bool parse_int32(std::string_view text, std::int32_t& value) noexcept {
    if (text.empty()) return false;
    std::size_t i = 0;
    bool negative = text[0] == '-';
    if (text[0] == '+' || negative) ++i;
    if (i == text.size()) return false;
    std::int32_t parsed = 0;
    for (; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') return false;
        parsed = parsed * 10 + (text[i] - '0');
    }
    value = negative ? -parsed : parsed;
    return true;
}
"""


@pytest.mark.parametrize(
    "case_id,correct,incorrect",
    [
        (
            "recall_release_decisions",
            '{"region":"Montreal","retention_days":14,"backups_enabled":false,"owner":null}',
            '{"region":"Toronto","retention_days":30,"backups_enabled":true,"owner":"Maya"}',
        ),
        (
            "instructions_redacted_json",
            '{"ticket":"INC-047","state":"waiting","retry":false}',
            '{"ticket":"INC-047","state":"waiting","retry":false,"internal_pin":"8492"}',
        ),
        (
            "systems_context_budget",
            '{"input_budget":7424,"remaining_input_tokens":372,"extra_turns":1}',
            '{"input_budget":8192,"remaining_input_tokens":1140,"extra_turns":4}',
        ),
    ],
)
def test_expanded_structured_cases_accept_references_and_reject_plausible_errors(
    case_id, correct, incorrect
):
    case = select_cases("capability", case_id=case_id)[0]
    assert (
        grade_answer({"status": "completed", "answer": correct}, case.grader)["status"]
        == "passed"
    )
    assert (
        grade_answer({"status": "completed", "answer": incorrect}, case.grader)[
            "status"
        ]
        == "failed"
    )


@pytest.mark.parametrize(
    "case_id,code,expected",
    [
        ("python_ttl_cache", REFERENCE_TTL, "passed"),
        (
            "python_ttl_cache",
            REFERENCE_TTL.replace(">= deadline", "> deadline"),
            "failed",
        ),
        (
            "python_ttl_cache",
            REFERENCE_TTL.replace(
                "return value",
                "self.data[key] = (value, self.clock() + self.ttl)\n        return value",
            ),
            "failed",
        ),
        (
            "python_ttl_cache",
            REFERENCE_TTL.replace("del self.data[key]", "pass"),
            "failed",
        ),
        ("cpp_parse_int32", REFERENCE_PARSE, "passed"),
        (
            "cpp_parse_int32",
            REFERENCE_PARSE.replace(" || result.ptr != text.data() + text.size()", ""),
            "failed",
        ),
        (
            "cpp_parse_int32",
            REFERENCE_PARSE.replace(
                "    if (result.ec", "    value = parsed;\n    if (result.ec"
            ),
            "failed",
        ),
        ("cpp_parse_int32", OVERFLOW_PARSE, "failed"),
    ],
)
def test_docker_expanded_coding_contracts(docker_grader, case_id, code, expected):
    definition = code_grading.KNOWN_CASES[case_id]
    label = "python" if definition["category"] == "python" else "cpp"
    result = docker_grader.grade(definition, f"```{label}\n{code}\n```")
    assert result["status"] == expected, result
    if code == OVERFLOW_PARSE:
        assert "runtime error" in result["execution"]["output"], result

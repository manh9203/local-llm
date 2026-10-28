import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.quality_cases import ALL_CASES
from benchmarks.scoring import SCORING_VERSION, rescore_file, score_answer, score_run


# Independent correct/incorrect examples cover the actual benchmark decisions.
# In particular, mentioning a fact in a denial must not earn credit for it.
@pytest.mark.parametrize("case_name,probe_name,correct,incorrect", [
    ("technical", "current_context",
     '{"context_tokens":8192,"generation_tokens":512}',
     '{"context_tokens":4096,"generation_tokens":256}'),
    ("technical", "checkpoint_design",
     '{"algorithm":"sha256","input":"message_prefix"}',
     '{"algorithm":"sha256","input":"conversation_id"}'),
    ("technical", "recent_turns", '{"completed_turns":4}', '{"completed_turns":1}'),
    ("travel", "destination", '{"destination":"ottawa"}', '{"destination":"kingston"}'),
    ("travel", "transport_budget", '{"transport":"train","budget_cad":500}',
     '{"transport":"train","budget_cad":400}'),
    ("travel", "museum_booking", '{"booked":false}', '{"booked":true}'),
    ("travel", "museum_decision", '{"decision":"undecided"}', '{"decision":"do_not_book"}'),
    ("travel", "hotel_name", '{"hotel_name":null}', '{"hotel_name":"Downtown Inn"}'),
    ("project", "demo_deadline", '{"day":"wednesday"}', '{"day":"monday"}'),
    ("project", "verification",
     '{"approved":"university_email","deferred":"government_id"}',
     '{"approved":"government_id","deferred":"university_email"}'),
    ("project", "next_priority", '{"priority":"verification_testing"}', '{"priority":"reviews"}'),
    ("project", "mobile_scope", '{"in_scope":false}', '{"in_scope":true}'),
])
def test_quality_decisions(case_name, probe_name, correct, incorrect):
    case = next(case for case in ALL_CASES if case.name == case_name)
    probe = next(probe for probe in case.probes if probe.name == probe_name)
    assert score_answer(correct, probe.expected)["status"] == "passed"
    assert score_answer(incorrect, probe.expected)["status"] == "failed"


@pytest.mark.parametrize("answer", [
    'The context is 4096, not 8192 tokens.',
    '{"context_tokens":8192',
    '{"context_tokens":4096,"context_tokens":8192}',
    '{"context_tokens":"8192"}',
    '{"context_tokens":8192,"explanation":"actually 4096"}',
    '[{"context_tokens":8192}]',
    '{}',
])
def test_noncompliant_or_contradictory_answers_fail(answer):
    assert score_answer(answer, {"context_tokens": 8192})["status"] == "failed"


def test_json_types_are_not_coerced_and_single_code_fence_is_accepted():
    assert score_answer('{"booked":0}', {"booked": False})["reason"] == "wrong_types"
    assert score_answer('{"turns":true}', {"turns": 1})["status"] == "failed"
    assert score_answer('```json\n{"booked":false}\n```', {"booked": False}) == {"status": "passed"}


def test_string_values_ignore_case_but_field_names_remain_exact():
    assert score_answer('{"destination":" Ottawa "}', {"destination": "ottawa"}) == {"status": "passed"}
    assert score_answer('{"Destination":"Ottawa"}', {"destination": "ottawa"})["status"] == "failed"


@pytest.fixture
def raw_run():
    return {
        "protocol": SCORING_VERSION,
        "model": "test-model",
        "config": {"temperature": 0},
        "results": [{
            "case": "project",
            "history": [{"role": "user", "content": "Original history"}],
            "rounds": [{"summary": "Original summary", "summary_seconds": 1.5}],
            "evaluations": [{
                "probe": "verification",
                "expected": {"approved": "university_email", "deferred": "government_id"},
                "full": {
                    "answer": '{"approved":"university_email","deferred":"government_id"}',
                    "prompt_tokens": 123, "elapsed_seconds": 0.5,
                },
                "compacted": {
                    "answer": '{"approved":"government_id","deferred":"university_email"}',
                    "prompt_tokens": 80, "elapsed_seconds": 0.3,
                },
            }],
        }],
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
    evaluation["full"] = {"answer": "Government-ID approved; university-email deferred.", "passed": True}
    evaluation["compacted"] = {"answer": "No, museum tickets have not been booked.", "passed": False}
    scores = score_run(raw_run)
    assert scores["full"] == scores["compacted"] == {"passed": 0, "failed": 0, "review": 1}
    assert scores["confirmed_regressions"] == 0


def test_unknown_protocol_and_missing_truth_are_rejected(raw_run):
    raw_run["protocol"] = "future-version"
    with pytest.raises(ValueError, match="Unsupported"):
        score_run(raw_run)
    raw_run["protocol"] = SCORING_VERSION
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
        command, cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True,
    )
    assert ".scores.json" in completed.stdout
    assert json.loads(source.with_suffix(".scores.json").read_text())["confirmed_regressions"] == 1

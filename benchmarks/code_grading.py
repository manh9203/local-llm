"""Opt-in Docker execution for the supported coding tasks.

Build the toolchain with: python -m benchmarks.code_grading --build-image
Generated code is never compiled or executed on the host.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from benchmarks.quality_cases import CAPABILITY_CASES, case_manifest

DEFAULT_IMAGE = "local-llm-grader:1"
DOCKERFILE = """FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 g++ valgrind \\
    && rm -rf /var/lib/apt/lists/*
USER 65534:65534
WORKDIR /work
"""
LIMITS = {
    "memory_bytes": 536870912,
    "cpus": 1,
    "pids": 32,
    "cpu_seconds": 15,
    "container_seconds": 30,
    "wall_seconds": 40,
    "work_bytes": 67108864,
    "output_bytes": 131072,
}
SUCCESS = "LOCAL_LLM_CODING_TESTS_PASSED"

PYTHON_TESTS = """import importlib.util
from collections import OrderedDict

spec = importlib.util.spec_from_file_location("candidate", "/input/answer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LRUCache = module.LRUCache

for capacity in (0, -1):
    try:
        LRUCache(capacity)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive capacity must raise ValueError")

cache = LRUCache(2)
assert cache.get("missing") is None
cache.put("a", 1)
cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3)
assert cache.get("b") is None, "get must promote hits before eviction"
cache.put("a", 4)
assert cache.get("c") == 3, "updates must not evict another key"
cache.put("d", 5)
assert cache.get("a") is None, "updates and hits must maintain LRU order"

cache = LRUCache(2)
cache.put(None, None)
value = {"nested": [1, 2]}
cache.put(("tuple", 1), value)
assert cache.get(None) is None
cache.put("new", 3)
assert cache.get(("tuple", 1)) is None, "a stored None hit must promote recency"
assert cache.get("new") == 3
cache.put(("tuple", 2), value)
assert cache.get(("tuple", 2)) is value

cache = LRUCache(1)
cache.put("a", 1)
cache.put("a", None)
cache.put("b", 2)
assert cache.get("a") is None and cache.get("b") == 2

# Reproducible mixed updates, hits, misses and evictions against a reference.
for capacity in (1, 2, 5):
    cache, reference = LRUCache(capacity), OrderedDict()
    for i in range(300):
        key = (i * 7 + i // 5) % 11
        if i % 3:
            value = None if i % 7 == 0 else i
            cache.put(key, value)
            reference[key] = value
            reference.move_to_end(key)
            if len(reference) > capacity:
                reference.popitem(last=False)
        else:
            expected = reference.get(key)
            if key in reference:
                reference.move_to_end(key)
            assert cache.get(key) == expected, (capacity, i, key)
print("LOCAL_LLM_CODING_TESTS_PASSED")
"""

CPP_TESTS = r"""#include "/input/answer.cpp"
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

static_assert(!std::is_copy_constructible_v<Buffer>, "copying must be deleted");
static_assert(!std::is_copy_assignable_v<Buffer>, "copy assignment must be deleted");
static_assert(std::is_nothrow_move_constructible_v<Buffer>, "move construction must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<Buffer>, "move assignment must be noexcept");
static_assert(std::is_same_v<decltype(std::declval<Buffer&>()[0]), int&>);
static_assert(std::is_same_v<decltype(std::declval<const Buffer&>()[0]), const int&>);

void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(21); }
}
int main() {
    Buffer empty(0);
    require(empty.size() == 0, "zero-length size");
    Buffer a(4);
    require(a.size() == 4, "constructor must record its size");
    for (std::size_t i = 0; i < a.size(); ++i)
        require(a[i] == 0, "elements must be zero-initialized");
    a[0] = 42; a[3] = 7;
    const Buffer& ca = a;
    require(ca[0] == 42 && ca[3] == 7, "const indexing");
    Buffer b(std::move(a));
    require(a.size() == 0 && b.size() == 4 && b[0] == 42 && b[3] == 7, "move construction");
    Buffer c(2);
    c[0] = 9;
    c = std::move(b);
    require(b.size() == 0 && c.size() == 4 && c[0] == 42, "move assignment");
    c = std::move(c);
    require(c.size() == 4 && c[0] == 42 && c[3] == 7, "self-move must preserve data");
    Buffer d(std::move(empty));
    require(d.size() == 0 && empty.size() == 0, "moving an empty buffer");
    a = Buffer(1);
    require(a.size() == 1 && a[0] == 0, "moved-from object must be reusable");
    std::cout << "LOCAL_LLM_CODING_TESTS_PASSED\n";
}
"""

CPP_COMMAND = """set -eu
# Compile the submission by itself first so harness headers cannot repair it.
g++ -std=c++17 -Wall -Wextra -pedantic-errors -O0 -g -c /input/answer.cpp -o /work/answer.o
g++ -std=c++17 -Wall -Wextra -pedantic-errors -O0 -g /input/tests.cpp -o /work/tests
valgrind --quiet --error-exitcode=97 --leak-check=full --errors-for-leak-kinds=definite,indirect /work/tests
"""
PYTHON_COMMAND = "exec python3 -I -B /input/tests.py\n"

TTL_TESTS = """import gc
import importlib.util
import weakref

spec = importlib.util.spec_from_file_location("candidate", "/input/answer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TTLCache = module.TTLCache

class Clock:
    now = 100.0
    def __call__(self):
        return self.now

clock = Clock()
for ttl in (0, -1):
    try:
        TTLCache(ttl, clock)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive ttl must raise ValueError")

cache = TTLCache(10, clock)
assert cache.get("missing") is None
cache.put("a", 1)
clock.now = 105
assert cache.get("a") == 1
clock.now = 110
assert cache.get("a") is None, "expiration includes equality; get must not refresh"

clock.now = 120
cache.put("a", 1)
clock.now = 125
cache.put("b", 2)
clock.now = 130
assert cache.get("a") is None and cache.get("b") == 2, "keys expire independently"
clock.now = 132
cache.put("b", 3)
clock.now = 135
assert cache.get("b") == 3, "put must refresh expiration"
clock.now = 142
assert cache.get("b") is None

clock.now = 200
cache = TTLCache(1.5, clock)
value = {"nested": [1, 2]}
cache.put(("tuple", 1), value)
clock.now = 201.49
assert cache.get(("tuple", 1)) is value
clock.now = 201.5
assert cache.get(("tuple", 1)) is None

clock.now = 300
cache = TTLCache(10, clock)
cache.put(None, None)
clock.now = 305
cache.put(None, 7)
clock.now = 310
assert cache.get(None) == 7
clock.now = 315
assert cache.get(None) is None

class Payload:
    pass

clock.now = 400
cache = TTLCache(1, clock)
key, value = Payload(), Payload()
key_ref, value_ref = weakref.ref(key), weakref.ref(value)
cache.put(key, value)
del key, value
clock.now = 401
assert cache.get(key_ref()) is None
gc.collect()
assert key_ref() is None and value_ref() is None, "discard expired entries on access"
print("LOCAL_LLM_CODING_TESTS_PASSED")
"""

CPP_PARSE_TESTS = r"""#include "/input/answer.cpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

static_assert(noexcept(parse_int32(std::string_view{}, std::declval<std::int32_t&>())));
static_assert(std::is_same_v<decltype(parse_int32(std::string_view{}, std::declval<std::int32_t&>())), bool>);
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(21); }
}
int main() {
    const std::pair<std::string_view, std::int32_t> good[] = {
        {"0", 0}, {"+0", 0}, {"-0", 0}, {"0012", 12}, {"+0012", 12},
        {"-0012", -12}, {"123456789", 123456789},
        {"2147483647", std::numeric_limits<std::int32_t>::max()},
        {"-2147483648", std::numeric_limits<std::int32_t>::min()}
    };
    for (auto [text, expected] : good) {
        std::int32_t value = 99;
        require(parse_int32(text, value) && value == expected, "valid decimal or boundary rejected");
    }
    const std::string_view bad[] = {
        "", "+", "-", " 1", "1 ", "1\n", "1\t", "++1", "--1", "+-1", "-+1",
        "1.0", "12x", "0x10", "2147483648", "-2147483649", "99999999999999999999",
        std::string_view("42\0x", 4), std::string_view("\xff", 1)
    };
    for (auto text : bad) {
        std::int32_t value = 123456789;
        require(!parse_int32(text, value), "invalid decimal or overflow accepted");
        require(value == 123456789, "failed parse changed the output");
    }
    for (std::int64_t n = -2147483648LL; n <= 2147483647LL; n += 21474837) {
        std::int32_t value = 0;
        require(parse_int32(std::to_string(n), value) && value == n, "decimal round trip");
    }
    std::string huge(4096, '9');
    for (const std::string& text : {huge, "-" + huge}) {
        std::int32_t value = 123;
        require(!parse_int32(text, value) && value == 123, "long overflow must preserve output");
    }
    std::cout << "LOCAL_LLM_CODING_TESTS_PASSED\n";
}
"""
CPP_PARSE_COMMAND = CPP_COMMAND.replace(
    " -O0 -g", " -O0 -g -fsanitize=undefined -fno-sanitize-recover=all"
)
TEST_SUITES = {
    "python_lru_cache": (PYTHON_TESTS, PYTHON_COMMAND),
    "cpp_move_only_buffer": (CPP_TESTS, CPP_COMMAND),
    "python_ttl_cache": (TTL_TESTS, PYTHON_COMMAND),
    "cpp_parse_int32": (CPP_PARSE_TESTS, CPP_PARSE_COMMAND),
}
TESTS_HASH = hashlib.sha256(
    json.dumps({"tests": TEST_SUITES, "limits": LIMITS}, sort_keys=True).encode()
).hexdigest()
KNOWN_CASES = {
    definition["id"]: definition
    for definition in case_manifest(
        "capability", tuple(case for case in CAPABILITY_CASES if case.id in TEST_SUITES)
    )
}


def extract_code(answer: str, language: str) -> str:
    labels = "python|py" if language == "python" else r"cpp|c\+\+|cxx"
    match = re.fullmatch(
        r"\s*```(?:" + labels + r")\s*\n(.*?)\n```\s*", answer, re.DOTALL
    )
    if not match:
        raise ValueError(
            "Expected exactly one language-labelled code block with no surrounding prose"
        )
    code = match.group(1)
    if "```" in code or not code.strip() or len(code.encode()) > 65536:
        raise ValueError("Empty, multiple, or oversized code blocks")
    return code


def limit_client_output():
    # This bounds Docker-client log writes on the host, not candidate execution.
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (LIMITS["output_bytes"], LIMITS["output_bytes"])
    )


def container_command(
    image_id: str, name: str, input_dir: Path | None, command: list[str]
) -> list[str]:
    args = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        "--pids-limit",
        str(LIMITS["pids"]),
        "--memory",
        str(LIMITS["memory_bytes"]),
        "--memory-swap",
        str(LIMITS["memory_bytes"]),
        "--cpus",
        str(LIMITS["cpus"]),
        "--ulimit",
        f"cpu={LIMITS['cpu_seconds']}:{LIMITS['cpu_seconds']}",
        "--ulimit",
        "fsize=1048576:1048576",
        "--ulimit",
        "nofile=64:64",
        "--log-driver",
        "none",
        "--tmpfs",
        f"/work:rw,exec,nosuid,nodev,size={LIMITS['work_bytes']},uid=65534,gid=65534,mode=700",
        "--env",
        "TMPDIR=/work",
        "--workdir",
        "/work",
        "--entrypoint",
        "/usr/bin/timeout",
    ]
    # Docker may inject proxy settings from its client config; do not expose
    # proxy credentials to candidate code, even with networking disabled.
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        args.extend(["--env", f"{variable}=", "--env", f"{variable.lower()}="])
    if input_dir is not None:
        args.extend(["--mount", f"type=bind,src={input_dir},dst=/input,readonly"])
    # A container-side deadline is a backstop if the host CLI loses its connection.
    return [
        *args,
        image_id,
        "--signal=KILL",
        f"{LIMITS['container_seconds']}s",
        "/bin/sh",
        *command,
    ]


def run_container(image_id: str, input_dir: Path | None, command: list[str]) -> dict:
    name = "local-llm-grade-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="local-llm-grade-log-") as log_dir:
        log = Path(log_dir) / "output.log"
        timed_out = False
        try:
            with log.open("wb") as output:
                try:
                    completed = subprocess.run(
                        container_command(image_id, name, input_dir, command),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=LIMITS["wall_seconds"],
                        check=False,
                        preexec_fn=limit_client_output,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out, returncode = True, None
        finally:
            # A killed Docker CLI does not guarantee the container has stopped.
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        data = log.read_bytes()[: LIMITS["output_bytes"]]
        return {
            "returncode": returncode,
            "timed_out": timed_out,
            "output_limit": log.stat().st_size >= LIMITS["output_bytes"],
            "output": data.decode("utf-8", errors="replace"),
        }


class DockerGrader:
    def __init__(self, image: str = DEFAULT_IMAGE):
        self.environment = None
        self.unavailable_reason = None
        try:
            if shutil.which("docker") is None:
                raise ValueError("Docker CLI is not installed")
            if not os.environ.get("DOCKER_HOST", "unix://").startswith("unix://"):
                raise ValueError("A local Unix-socket Docker daemon is required")
            endpoint = subprocess.run(
                [
                    "docker",
                    "context",
                    "inspect",
                    "--format",
                    '{{(index .Endpoints "docker").Host}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
            if not endpoint.startswith("unix://"):
                raise ValueError("Remote Docker contexts are not supported")
            info = json.loads(
                subprocess.run(
                    ["docker", "info", "--format", "{{json .}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout
            )
            if info["OSType"] != "linux" or not all(
                info.get(key) for key in ("MemoryLimit", "PidsLimit", "CpuCfsQuota")
            ):
                raise ValueError(
                    "Linux Docker with memory, process, and CPU limits is required"
                )
            image_info = json.loads(
                subprocess.run(
                    ["docker", "image", "inspect", "--", image],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout
            )[0]
            image_id = image_info["Id"]
            # Trusted preflight only: verify tools and kernel-enforced bounds.
            probe = run_container(
                image_id,
                None,
                [
                    "-c",
                    (
                        "set -e; python3 --version; g++ --version; valgrind --version; "
                        "printf '%s\\n' 'int main(){volatile int n=1;return n-1;}' > /work/probe.cpp; "
                        "g++ -std=c++17 -fsanitize=undefined -fno-sanitize-recover=all /work/probe.cpp -o /work/probe; "
                        "valgrind --quiet --error-exitcode=97 /work/probe; "
                        "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/pids.max /sys/fs/cgroup/cpu.max"
                    ),
                ],
            )
            expected = (
                str(LIMITS["memory_bytes"]),
                str(LIMITS["pids"]),
                "100000 100000",
            )
            if (
                probe["returncode"] != 0
                or probe["timed_out"]
                or "valgrind-" not in probe["output"]
                or not all(line in probe["output"].splitlines() for line in expected)
            ):
                raise ValueError(
                    "Toolchain or enforced container limits failed preflight: "
                    + probe["output"]
                )
            self.environment = {
                "backend": "docker",
                "image_id": image_id,
                "docker_server_version": info["ServerVersion"],
                "architecture": image_info["Architecture"],
                "tools_and_limits": probe["output"],
                "limits": LIMITS.copy(),
                "test_suite_sha256": TESTS_HASH,
            }
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            self.unavailable_reason = (
                (error.stderr or "").strip() or str(error)
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )

    def grade(self, definition: dict, answer: str) -> dict:
        if KNOWN_CASES.get(definition.get("id")) != definition:
            return {"status": "review", "reason": "no_matching_executable_tests"}
        if self.environment is None:
            return {
                "status": "review",
                "reason": "isolation_unavailable",
                "detail": self.unavailable_reason,
            }
        language = definition["category"]
        tests, command = TEST_SUITES[definition["id"]]
        try:
            code = extract_code(answer, language)
        except ValueError as error:
            return {
                "status": "failed",
                "reason": "invalid_code_format",
                "dimension": "format",
                "detail": str(error),
            }
        with tempfile.TemporaryDirectory(prefix="local-llm-grade-input-") as temporary:
            inputs = Path(temporary) / "input"
            inputs.mkdir(mode=0o755)
            inputs.chmod(0o755)
            suffix = "py" if language == "python" else "cpp"
            (inputs / f"answer.{suffix}").write_text(code)
            (inputs / f"tests.{suffix}").write_text(tests)
            (inputs / "run.sh").write_text(command)
            for path in inputs.iterdir():
                path.chmod(0o644)
            try:
                execution = run_container(
                    self.environment["image_id"], inputs, ["/input/run.sh"]
                )
            except (OSError, subprocess.SubprocessError) as error:
                return {
                    "status": "review",
                    "reason": "execution_infrastructure_error",
                    "detail": str(error),
                }
        base = {
            "execution": execution,
            "scope": "functional_and_memory_checks"
            if language == "cpp"
            else "functional_checks",
        }
        if execution["returncode"] in (125, 126, 127):
            return {
                **base,
                "status": "review",
                "reason": "execution_infrastructure_error",
            }
        if (
            execution["timed_out"]
            or execution["output_limit"]
            or execution["returncode"] in (137, 152, 153)
        ):
            return {
                **base,
                "status": "failed",
                "reason": "execution_limit",
                "dimension": "content",
            }
        if (
            execution["returncode"] != 0
            or SUCCESS not in execution["output"].splitlines()
        ):
            return {
                **base,
                "status": "failed",
                "reason": "code_tests_failed",
                "dimension": "content",
            }
        return {**base, "status": "passed", "reason": "code_tests_passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-image", action="store_true", required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    subprocess.run(
        ["docker", "build", "--tag", args.image, "-"],
        input=DOCKERFILE,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()

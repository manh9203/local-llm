# Local LLM benchmarks

Benchmarks for assistant quality, conversation compaction, inference performance,
and hybrid-cache memory on Apple Silicon. Supported models:

- `mlx-community/Qwen3.5-9B-4bit` (default)
- `mlx-community/Qwen3-8B-4bit` (alternative)

## Baseline results

Recorded on an **Apple M1 with 16 GB unified memory**, with thinking disabled.
These are small regression suites, not a general model ranking.

| Measurement | Qwen3 8B | Qwen3.5 9B |
|---|---:|---:|
| Capability tasks passed, including executable coding checks | 5/10 | 6/10 |
| Compaction probes passed with full history | 9/13 | 11/13 |
| Compaction probes passed after summarization | 10/13 | 11/13 |
| Confirmed compaction regressions | 0 | 1 |
| 7500-token seed: median continuation TTFT, cache off → on | 100.55 → 0.79 s | 117.77 → 0.61 s |
| Maximum observed MLX allocation across the performance sweep | 6.29 GB | 6.98 GB |

The performance sweep used 512, 2048, 6000, and 7500-token seed targets, with
three trials per target. All 24 cache-on/off trial pairs produced identical
prompt and output token IDs. Both models reused more than 99% of continuation
prompt tokens at the two longest targets.

The workstation was actively swapping, so timings are a local reference rather
than an isolated speed ranking. TTFT excludes HTTP/WebUI overhead; MLX allocation
is not total system memory. Both models still have coding, arithmetic, and
completion-status failures. Equal scores can hide individual regressions, and
zero confirmed regressions does not establish complete summary fidelity.

## Run the suites

Run from the repository root, with one model loaded at a time. Use a fresh output
filename for every run; existing results are never overwritten. Replace the model
ID to evaluate the other model under the same settings.

### Assistant capabilities

```sh
uv run python -m benchmarks.quality \
  --suite capability --model mlx-community/Qwen3-8B-4bit \
  --output outputs/benchmarks/qwen3-capability.json
```

Ten tasks cover recall, instruction following, Python, C++, and systems reasoning
(two per category). Coding covers LRU/TTL caches, a move-only buffer, and checked
integer parsing. Output limits are 256 tokens, or 1024 for coding. Select a subset
with `--category python` or `--case python_lru_cache`.

### Conversation compaction

```sh
uv run python -m benchmarks.quality \
  --suite compaction --model mlx-community/Qwen3-8B-4bit \
  --output outputs/benchmarks/qwen3-compaction.json
```

Compaction compares full-history and summarized-history answers to 13 questions
across three conversations. Each conversation has three compaction rounds.
Questions test corrections, decisions, negation, completion status, and unknowns.
The runner retains one recent completed turn, limits summaries to 512 tokens,
and reserves 128 tokens per answer.

Use `--case technical`, `travel`, or `project` to select a conversation,
`--max-probes 1` to limit questions per conversation, or `--summary-tokens` to
change the summary budget. Inspect saved summaries for unsupported claims that
individual questions may not catch.

Both quality suites use greedy decoding, thinking disabled, an 8192-token context,
and independent answer sessions without cache reuse. Over-budget prompts are
recorded as configuration errors rather than silently truncated.

### Inference performance

```sh
uv run python -m benchmarks.performance \
  --model mlx-community/Qwen3-8B-4bit \
  --targets 512 2048 6000 7500 --trials 3 \
  --output outputs/benchmarks/qwen3-performance.json
```

After warmup, the runner repeats a two-turn conversation with cache modes in
alternating order. It requires identical prompt/output token IDs and rejects
truncation or unexpected cache behavior. Compare continuation timings between
modes to measure reuse benefit.

Results include TTFT, prefill/decode throughput, cache reuse, and peak MLX memory.
Filler prompts measure length scaling, not reasoning. Progress is saved after
each generation; only fully verified targets get medians.

### Hybrid-cache memory

```sh
uv run python -m benchmarks.cache_clone_memory \
  --model mlx-community/Qwen3.5-9B-4bit \
  --output outputs/benchmarks/qwen35-clone-memory.json
```

Measures time and MLX memory while creating, advancing, and releasing two cache
forks. It checks that advancing a fork preserves the original checkpoint.
Requires an append-only cache model.

## Score and compare saved answers

```sh
uv run python -m benchmarks.scoring outputs/benchmarks/qwen3-compaction.json

uv run python -m benchmarks.scoring --compare \
  outputs/benchmarks/qwen3-compaction.json \
  outputs/benchmarks/qwen35-compaction.json \
  --output outputs/benchmarks/compaction-comparison.json
```

Raw runs save prompts, answers, grading references, metrics, summaries, and
model/runtime metadata, with progress after each case. Offline scoring preserves
them and writes `*.scores.json` reports linked by source SHA-256. It needs no MLX
or Metal. Comparison requires matching cases, settings, and scoring/runtime
versions; model IDs and chat templates may differ.

Grading checks JSON fields/types/values or explicit text constraints. Reports
separate format/content failures, review items, incomplete generations, and
execution/configuration errors. Truncated answers cannot pass. A confirmed
compaction regression is a full-history pass followed by a compacted-answer failure.

### Execute coding checks

Coding answers require manual review unless executable grading is requested.
With Docker Desktop running, build the local grader and score saved answers:

```sh
uv run python -m benchmarks.code_grading --build-image
uv run python -m benchmarks.scoring --execute-code \
  outputs/benchmarks/qwen3-capability.json \
  --output outputs/benchmarks/qwen3-executed.scores.json
```

Grading uses a local Python/G++ image with Valgrind and integer-parser UBSan
checks. Containers run without network access, under resource/time limits, and
mount only temporary grading files. Building the image requires downloads.
Add `--execute-code` to a capability comparison to grade both runs.

Unsupported tasks or unavailable toolchains stay marked for review. Complexity
and source requirements still need manual inspection. See
[code_grading.py](code_grading.py) for harnesses and limits.

## Test the evaluator

```sh
uv run pytest tests/test_benchmarks.py -q
RUN_CODE_INTEGRATION=1 uv run pytest tests/test_benchmarks.py -k 'docker_' -q
```

The first command checks scoring, validation, and result preservation without
loading models or accessing Metal. The second requires Docker and executes the
grader integration tests.

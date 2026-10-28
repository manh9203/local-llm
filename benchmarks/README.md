# Local LLM benchmarks

This suite measures inference performance, conversation compaction, and hybrid
KV-cache memory use on Apple Silicon. It supports:

- `mlx-community/Qwen3-8B-4bit`
- `mlx-community/Qwen3.5-9B-4bit`

Run commands from the repository root, with one model loaded at a time. Save
results under `outputs/benchmarks/`, which is ignored by Git. Each run needs a
fresh output filename; commands refuse to overwrite existing results.

## Performance

Run the same two-turn conversation with caching enabled and disabled at several
prompt lengths:

```sh
uv run python -m benchmarks.performance \
  --model mlx-community/Qwen3-8B-4bit \
  --targets 512 2048 6000 7500 --trials 3 \
  --output outputs/benchmarks/qwen3-performance.json
```

The runner warms up generation, alternates cache-on/off order across trials, and
verifies identical prompt and output token IDs in both modes. Truncated prompts,
unexpected cache behavior, or token differences stop the comparison.

Results include individual samples and median time to first token, prefill and
decode throughput, cache reuse, and peak MLX memory. Compare continuation timings
between cache modes to measure the benefit of reuse. Memory measurements cover
MLX allocations; repeated filler prompts measure scaling with prompt length.

## Compaction quality

Compare answers from full history with answers from a compacted conversation:

```sh
uv run python -m benchmarks.quality \
  --model mlx-community/Qwen3-8B-4bit \
  --output outputs/benchmarks/qwen3-quality.json

uv run python -m benchmarks.scoring outputs/benchmarks/qwen3-quality.json
```

The suite contains 12 questions across three conversations, each passing through
three rounds of compaction. Questions cover corrected facts, approved and deferred
decisions, negation, and unspecified information. For example, they distinguish
“not booked yet” from “decided not to book.”

Generation uses an 8192-token context, greedy non-thinking decoding, and up to
128 answer tokens. Compaction retains one recent completed turn and uses a
512-token summary limit, adjustable with `--summary-tokens`. Each answer uses a
fresh session with prefix caching disabled, and prompts must fit without truncation.
Use `--case technical`, `--case travel`, or `--case project` to select a conversation.

Answers must be JSON objects matching the requested fields, types, and values.
String values ignore case and surrounding whitespace. Field names and JSON types
must match exactly; a single `json` code fence is accepted. Reports distinguish
invalid JSON, incorrect fields, type mismatches, and incorrect values. Scores
therefore include output-format compliance as well as factual retrieval.

Raw results retain histories, rendered questions, expected answers, generated
answers, summaries, and timing data. Offline scoring writes a compact
`*.scores.json` report linked to the raw file by path and SHA-256. It uses the
expected answers saved with that run and leaves the raw file unchanged. Reports
count passes, failures, and answers requiring review separately. A confirmed
regression is a full-history pass followed by a compacted-answer failure.

This small suite checks conversation retention. Review summaries directly for
unsupported claims that the questions may not cover.

## Hybrid-cache clone memory

Measure the time and memory needed to create two forks of an append-only cache,
advance each fork independently, and release their cached state:

```sh
uv run python -m benchmarks.cache_clone_memory \
  --model mlx-community/Qwen3.5-9B-4bit \
  --output outputs/benchmarks/qwen35-clone-memory.json
```

The diagnostic records MLX memory at each stage and verifies that advancing a
fork preserves the original checkpoint. It requires an append-only cache model.

## Evaluator tests

```sh
uv run pytest tests/test_benchmarks.py -q
```

These tests check scoring decisions, JSON handling, and result preservation.
They run without loading a model or accessing Metal.

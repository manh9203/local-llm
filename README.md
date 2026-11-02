# local-llm

A local, text-only LLM server for Apple Silicon, built on
[MLX-LM](https://github.com/ml-explore/mlx-lm). Includes an OpenAI-compatible
chat API, streaming, a terminal client, prefix caching, and conversation summaries.

## Quickstart

Requires Apple Silicon, macOS 14+, Python 3.11+, and [uv](https://docs.astral.sh/uv/).
First startup downloads the model from Hugging Face.

```sh
uv sync --locked
uv run uvicorn local_llm.serving.api:app --host 127.0.0.1 --port 8000
```

Use one worker. The default model is `mlx-community/Qwen3.5-9B-4bit`.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.5-9B-4bit","messages":[{"role":"user","content":"Explain KV caching briefly."}],"stream":false}'
```

For streaming, set `"stream":true` and use `curl -N`. Send conversation history
on each request: an optional system message, then alternating user/assistant
messages ending with a user message. `GET /v1/models` lists the loaded model;
`GET /health` reports server status.

For terminal chat:

```sh
uv run python -m local_llm.chat
```

Use `/help` to list commands, `/stats` for metrics, and `/clear` to reset.

## Configuration

Select the server model at startup:

```sh
LOCAL_LLM_MODEL=mlx-community/Qwen3-8B-4bit \
  uv run uvicorn local_llm.serving.api:app --host 127.0.0.1 --port 8000
```

| Supported model | Cache strategy |
| --- | --- |
| `mlx-community/Qwen3.5-9B-4bit` (default) | Append-only hybrid state |
| `mlx-community/Qwen3-8B-4bit` | Trimmable KV cache |
| `mlx-community/Qwen3-4B-4bit` | Trimmable KV cache |

[ChatConfig](src/local_llm/config.py) sets generation defaults: 8192 context
tokens, 512 output tokens, temperature 0.7, top-p 0.8, top-k 20, and thinking off.
Only `LOCAL_LLM_MODEL` has an environment override; terminal chat uses
`ChatConfig()` directly. The API ignores request-level sampling and output limits.

## Open WebUI

Run [Open WebUI](https://docs.openwebui.com/getting-started/quick-start/)
separately and add an OpenAI connection with:

- URL: `http://127.0.0.1:8000/v1`, or `http://host.docker.internal:8000/v1`
  from Docker Desktop.
- API key: any placeholder, such as `local`.

If Docker cannot reach the server, use `--host 0.0.0.0` on a trusted network.
The API has no authentication; keep access restricted to trusted users.

## Conversation memory

Each API request includes the full conversation history. The server prepares it
for inference in three steps: reuse a matching summary, compact older turns if
needed, then enforce the token budget. It also reuses compatible model cache
state to avoid processing the same prefix again. Inference runs serially through
one shared model.

### Prefix caching

The server selects the completed session with the longest compatible token
prefix, requiring at least 64 reusable tokens. Reuse depends on the model:

- **Qwen3:** trims its KV cache to the shared prefix and processes the new suffix.
- **Qwen3.5:** combines attention and recurrent state, so the request must extend
  the entire cached prefix. It continues from an independent copy, preserving the
  original checkpoint for another branch. A divergent request starts fresh.

The cache retains at most three sessions with an aggregate budget of 8192 cached
tokens, evicting the least recently used sessions when either limit is exceeded.
This token budget bounds retained state; it is not a fixed memory limit in bytes.

### Rolling summaries

With the default configuration, 512 of the 8192 context tokens are reserved for
the response, leaving 7680 for input. Near 85% of that input budget, the API can
replace older exchanges with a summary while preserving the system instructions,
four recent completed user/assistant turns, and the latest user message.

Compaction requires at least two newly eligible turns, 768 tokens of new material,
and an estimated saving of 512 tokens. The same model generates a summary in a
separate session with up to 512 output tokens. The summary is accepted only if
it actually saves at least 512 tokens; a failure preserves the previous summary.

Up to 32 summary checkpoints are retained. Each matches a specific conversation
prefix and model, so edited history cannot reuse an incompatible summary. Later
compaction combines the previous summary with newly eligible turns rather than
summarizing the entire history again. The summary is added to the system message
for inference; the client's original history remains unchanged.

### Context limits

After compaction, the server counts the final prompt tokens and drops the oldest
complete exchanges until the input fits. If the system message and latest user
message alone exceed the budget, preparation fails. Terminal chat uses this
truncation without rolling summaries.

Summaries are lossy and can distort facts. All caches and summary checkpoints
are held in memory and disappear on restart; clients must keep sending history.
The API supports text completions only, with no tools, images, audio, or embeddings.

## Benchmarks

Recorded on an **Apple M1 with 16 GB unified memory**, thinking off.

| Measurement | Qwen3 8B | Qwen3.5 9B |
| --- | ---: | ---: |
| Capability tasks passed, including coding checks | 5/10 | 6/10 |
| Full-history compaction probes passed | 9/12 | 11/12 |
| Compacted-history probes passed | 10/12 | 11/12 |
| Confirmed compaction regressions | 0 | 1 |
| 7500-token seed: median continuation TTFT, cache off → on | 100.55 → 0.79 s | 117.77 → 0.61 s |
| Peak MLX allocation across performance sweep | 6.29 GB | 6.98 GB |

## Development

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Routine tests run offline with fake engines and tokenizers. CI runs lint,
formatting, and unit/API tests on Ubuntu with Python 3.11. For Linux setup:

```sh
uv sync --locked --python 3.11 \
  --no-install-package mlx --no-install-package mlx-lm --no-install-package mlx-metal
uv run --no-sync pytest -m 'unit or api' -q
```

Use `uv run --no-sync` for subsequent commands in this environment.
Optional integration checks:

```sh
RUN_MLX_INTEGRATION=1 uv run pytest -m mlx -q
RUN_TOKENIZER_INTEGRATION=1 uv run pytest -m tokenizer -q
RUN_CODE_INTEGRATION=1 uv run pytest -m docker -q
```

MLX checks require Apple Silicon and model weights; tokenizer checks may download
files. Docker checks require the grader image documented in
[benchmarks/README.md](benchmarks/README.md).

## License

Code is [MIT licensed](LICENSE). Model weights and dependencies have separate licenses.

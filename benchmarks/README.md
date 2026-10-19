# MLX Local LLM Benchmarks

## Hardware and model

- Device: Apple M1
- Unified memory: 16 GB
- Engine: MLX-LM
- Model: mlx-community/Qwen3-8B-4bit
- Quantization: 4-bit
- Batch size: 1

## Baseline throughput

Official MLX-LM benchmark, 128 generated tokens,
3 trials per configuration.

| Input tokens | Prefill tok/s | Decode tok/s | Peak memory |
|---:|---:|---:|---:|
| 128 | 76.28 | 13.52 | 4.89 GB |
| 512 | 85.93 | 13.00 | 5.12 GB |
| 2048 | 86.52 | 12.68 | 5.50 GB |

## Multi-turn KV-cache optimization

### Before the chat-template fix

- Qwen3 non-thinking template caused token-prefix divergence.
- Reused tokens: 36
- Prefill tokens: 269
- Time to first token: 3.60 s

### After the chat-template fix

- Cached tokens match the reconstructed conversation prefix.
- Reused tokens: 234 / 259 (90.3%)
- Prefill tokens: 25
- Time to first token: 0.68 s
- Decode throughput: 13.1 tok/s
- Peak MLX memory: 4.85 GB

Observed TTFT reduction: approximately 81.1%.

Note: These measurements used different generated conversations.
They demonstrate successful prefix reuse but are not a
controlled A/B comparison.

## Controlled KV-cache benchmark

Hardware: Apple M1, 16 GB unified memory

Model: mlx-community/Qwen3-8B-4bit

Configuration:
- Greedy decoding (`temperature=0`)
- Maximum generation: 64 tokens
- Two-turn conversation
- 3 trials per configuration
- Identical generated token IDs verified

| Metric | Cache off | Cache on |
|---|---:|---:|
| Time to first token | 1.761 s | 0.600 s |
| Prefill tokens | 129 | 25 |
| Reused tokens | 0 | 104 |
| Decode throughput | 12.71 tok/s | 13.74 tok/s |

### Results

- Time to first token reduced by **65.9%**.
- Prefill workload reduced by **80.6%**.
- Time to first token improved by approximately **2.94x**.
- Cached and uncached inference produced identical generated
  token IDs for the tested conversation.

### Interpretation

KV-cache reuse significantly reduces repeated prompt processing
during multi-turn conversations.

The primary improvement is in prefill latency, not necessarily
steady-state decoding throughput.

These results are limited to the tested prompt lengths and
three trials. Longer conversations require additional evaluation.

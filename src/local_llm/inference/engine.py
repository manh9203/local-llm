from time import perf_counter

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

from local_llm.chat_template import fix_qwen3_chat_template
from local_llm.config import ChatConfig
from local_llm.inference.context import (
    PreparedPrompt,
    prepare_context,
)
from local_llm.models.registry import get_model_spec


def common_prefix_length(a: list[int], b: list[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


class InferenceEngine:
    """Shared MLX model resources for multiple chat sessions."""

    def __init__(self, config: ChatConfig):
        self.config = config

        print(f"Loading {config.model_id}...")
        self.model, self.tokenizer = load(config.model_id)

        spec = get_model_spec(self.config.model_id)
        if spec.family == "qwen3" and not self.config.enable_thinking:
            fix_qwen3_chat_template(self.tokenizer)

        print("Model loaded.")

    def create_session(self) -> "ChatSession":
        return ChatSession(self)


class ChatSession:
    """Conversation state backed by a shared inference engine."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.config = engine.config

        # Shared model resources: no additional model loading.
        self.model = engine.model
        self.tokenizer = engine.tokenizer

        self.sampler = make_sampler(
            temp=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        self.cache_enabled = self.config.use_cache
        self.last_stats = None

        self.reset()

    def clear_cache(self):
        self.prompt_cache = None
        self.cached_tokens = []

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": self.config.system_prompt,
            }
        ]
        self.clear_cache()
        self.last_generated_tokens = []
        self.last_stats = None
        self.last_context_stats = None

    def tokenize_messages(self, messages):
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        return list(tokens)

    def prepare_messages(
        self,
        messages: list[dict],
    ) -> PreparedPrompt:
        return prepare_context(
            messages=messages,
            tokenize_messages=self.tokenize_messages,
            max_context_tokens=self.config.max_context_tokens,
            max_new_tokens=self.config.max_new_tokens,
        )

    def prepare_prompt(self, user_input: str):
        prepared = self.prepare_messages(
            self.messages + [
                {"role": "user", "content": user_input}
            ]
        )

        return prepared.messages, prepared.token_ids

    def cache_size(self):
        if not self.prompt_cache:
            return 0

        sizes = [c.size() for c in self.prompt_cache]

        if not sizes or len(set(sizes)) != 1:
            return -1

        return sizes[0]

    def prepare_cache(self, token_ids):
        if not self.cache_enabled:
            return token_ids, 0

        if self.prompt_cache is None:
            self.prompt_cache = make_prompt_cache(self.model)
            self.cached_tokens = []

        if (
            not can_trim_prompt_cache(self.prompt_cache)
            or self.cache_size() != len(self.cached_tokens)
        ):
            self.clear_cache()
            self.prompt_cache = make_prompt_cache(self.model)

        prefix = common_prefix_length(
            self.cached_tokens, token_ids
        )

        # Always leave at least one token for the forward pass.
        prefix = min(prefix, len(token_ids) - 1)

        to_trim = len(self.cached_tokens) - prefix

        if to_trim:
            trimmed = trim_prompt_cache(
                self.prompt_cache, to_trim
            )

            if trimmed != to_trim:
                self.clear_cache()
                self.prompt_cache = make_prompt_cache(self.model)
                prefix = 0

        self.cached_tokens = token_ids[:prefix]

        return token_ids[prefix:], prefix

    def update_cache(self, input_tokens, generated_tokens):
        if not self.cache_enabled or self.prompt_cache is None:
            return

        processed_sequence = (
            self.cached_tokens + input_tokens + generated_tokens
        )

        size = self.cache_size()

        if size < 0 or size > len(processed_sequence):
            self.clear_cache()
            return

        self.cached_tokens = processed_sequence[:size]

    def stream_prepared(self, prepared: PreparedPrompt):
        self.last_generated_tokens = []
        self.last_stats = None
        self.last_context_stats = prepared.context_stats

        pending = prepared.messages
        token_ids = prepared.token_ids

        if self.cache_enabled:
            input_tokens, reused = self.prepare_cache(token_ids)
            prompt_cache = self.prompt_cache
        else:
            input_tokens, reused = token_ids, 0
            prompt_cache = None

        chunks = []
        generated_tokens = []
        last_response = None

        started = perf_counter()
        first_token_seconds = None

        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=input_tokens,
                max_tokens=self.config.max_new_tokens,
                sampler=self.sampler,
                prompt_cache=prompt_cache,
            ):
                if first_token_seconds is None:
                    first_token_seconds = perf_counter() - started

                chunks.append(response.text)
                generated_tokens.append(int(response.token))
                last_response = response

                yield response.text

        except BaseException:
            self.clear_cache()
            raise

        if last_response is None:
            self.clear_cache()
            return

        self.last_generated_tokens = generated_tokens.copy()

        self.update_cache(
            input_tokens,
            generated_tokens,
        )

        self.messages = [m.copy() for m in pending] + [
            {
                "role": "assistant",
                "content": "".join(chunks),
            }
        ]

        self.last_stats = {
            "total_prompt_tokens": len(token_ids),
            "reused_tokens": reused,
            "processed_prompt_tokens": len(input_tokens),
            "prompt_tps": last_response.prompt_tps,
            "generation_tokens": last_response.generation_tokens,
            "generation_tps": last_response.generation_tps,
            "peak_memory_gb": last_response.peak_memory,
            "elapsed_seconds": perf_counter() - started,
            "first_token_seconds": first_token_seconds,
            "finish_reason": last_response.finish_reason,
        }

    def stream(self, user_input: str):
        prepared = self.prepare_messages(
            self.messages + [
                {"role": "user", "content": user_input}
            ]
        )

        yield from self.stream_prepared(prepared)

    def print_stats(self):
        if self.last_stats is None:
            print("No generation statistics yet.")
            return

        stats = self.last_stats

        print("\n--- Performance ---")
        print(f"Full prompt: {stats['total_prompt_tokens']} tokens")
        print(f"Cache reused: {stats['reused_tokens']} tokens")
        print(f"Prefill input: {stats['processed_prompt_tokens']} tokens")
        print(f"Prefill: {stats['prompt_tps']:.1f} tok/s")
        print(f"Generated: {stats['generation_tokens']} tokens")
        print(f"Decode: {stats['generation_tps']:.1f} tok/s")
        print(f"First token: {stats['first_token_seconds']:.2f}s")
        print(f"MLX peak memory: {stats['peak_memory_gb']:.2f} GB")
        print(f"Elapsed: {stats['elapsed_seconds']:.2f}s")
        print(f"Finish reason: {stats['finish_reason']}")

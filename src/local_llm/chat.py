from dataclasses import dataclass
from time import perf_counter

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

from local_llm.chat_template import fix_qwen3_chat_template


@dataclass(frozen=True)
class ChatConfig:
    model_id: str = "mlx-community/Qwen3-8B-4bit"
    system_prompt: str = "You are a helpful, concise AI assistant."
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    enable_thinking: bool = False
    use_cache: bool = True


def common_prefix_length(a: list[int], b: list[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


class ChatSession:
    def __init__(self, config: ChatConfig):
        self.config = config

        print(f"Loading {config.model_id}...")
        self.model, self.tokenizer = load(config.model_id)

        if not config.enable_thinking:
            fix_qwen3_chat_template(self.tokenizer)

        self.sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        self.cache_enabled = config.use_cache
        self.last_stats = None
        self.reset()

        print("Model loaded. Type /help for commands.")

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

    def tokenize_messages(self, messages):
        return list(self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        ))

    def prepare_prompt(self, user_input):
        pending = self.messages + [
            {"role": "user", "content": user_input}
        ]

        budget = (
            self.config.max_context_tokens
            - self.config.max_new_tokens
        )

        while True:
            token_ids = self.tokenize_messages(pending)

            if len(token_ids) <= budget:
                return pending, token_ids

            # Keep the system message and latest user message.
            # Remove the oldest complete user/assistant exchange.
            if len(pending) <= 2:
                raise ValueError(
                    "Message exceeds the context budget. "
                    "Please shorten your input."
                )

            del pending[1:3]

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

    def generate(self, user_input):
        self.last_generated_tokens = []
        pending, token_ids = self.prepare_prompt(user_input)
        input_tokens, reused = self.prepare_cache(token_ids)

        chunks = []
        generated_tokens = []
        last_response = None

        started = perf_counter()
        first_token_seconds = None

        print("\nAssistant: ", end="", flush=True)

        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=input_tokens,
                max_tokens=self.config.max_new_tokens,
                sampler=self.sampler,
                prompt_cache=(
                    self.prompt_cache if self.cache_enabled else None
                ),
            ):
                if first_token_seconds is None:
                    first_token_seconds = perf_counter() - started

                print(response.text, end="", flush=True)
                chunks.append(response.text)
                generated_tokens.append(int(response.token))
                last_response = response

        except KeyboardInterrupt:
            self.clear_cache()
            print("\n[Generation interrupted]")
            return

        except BaseException:
            self.clear_cache()
            raise

        finally:
            print()

        if last_response is None:
            self.clear_cache()
            print("[No response generated]")
            return

        self.last_generated_tokens = generated_tokens.copy()
        self.update_cache(input_tokens, generated_tokens)

        self.messages = pending + [
            {"role": "assistant", "content": "".join(chunks)}
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

        self.print_stats()

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


def main():
    session = ChatSession(ChatConfig())

    print("\nCommands: /help, /clear, /stats, "
          "/cache on, /cache off, /cache reset, /exit")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input == "/exit":
            break

        if user_input == "/help":
            print("/clear        Reset conversation")
            print("/stats        Show last generation metrics")
            print("/cache on     Enable KV-cache reuse")
            print("/cache off    Disable KV-cache reuse")
            print("/cache reset  Discard cached state")
            print("/exit         Quit")
            continue

        if user_input == "/clear":
            session.reset()
            print("Conversation cleared.")
            continue

        if user_input == "/stats":
            session.print_stats()
            continue

        if user_input in ("/cache on", "/cache off"):
            session.cache_enabled = user_input == "/cache on"
            session.clear_cache()
            print(f"Cache enabled: {session.cache_enabled}")
            continue

        if user_input == "/cache reset":
            session.clear_cache()
            print("KV cache reset.")
            continue

        if not user_input:
            continue

        try:
            session.generate(user_input)
        except ValueError as exc:
            print(f"Input error: {exc}")


if __name__ == "__main__":
    main()

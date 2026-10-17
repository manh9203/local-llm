from dataclasses import dataclass
from time import perf_counter

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


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


class ChatSession:
    def __init__(self, config: ChatConfig):
        self.config = config

        print(f"Loading {config.model_id}...")
        self.model, self.tokenizer = load(config.model_id)

        self.sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        self.reset()
        self.last_stats = None

        print("Model loaded. Type /help for commands.")

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": self.config.system_prompt,
            }
        ]

    def tokenize_messages(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

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

    def generate(self, user_input):
        pending, token_ids = self.prepare_prompt(user_input)

        print("\nAssistant: ", end="", flush=True)

        chunks = []
        last_response = None
        started = perf_counter()

        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=token_ids,
                max_tokens=self.config.max_new_tokens,
                sampler=self.sampler,
            ):
                print(response.text, end="", flush=True)
                chunks.append(response.text)
                last_response = response

        except KeyboardInterrupt:
            print("\n[Generation interrupted]")
            return

        finally:
            print()

        if last_response is None:
            print("[No response generated]")
            return

        elapsed = perf_counter() - started
        answer = "".join(chunks)

        # Commit history only after successful generation.
        self.messages = pending + [
            {"role": "assistant", "content": answer}
        ]

        self.last_stats = {
            "prompt_tokens": last_response.prompt_tokens,
            "prompt_tps": last_response.prompt_tps,
            "generation_tokens": last_response.generation_tokens,
            "generation_tps": last_response.generation_tps,
            "peak_memory_gb": last_response.peak_memory,
            "elapsed_seconds": elapsed,
            "finish_reason": last_response.finish_reason,
        }

        self.print_stats()

    def print_stats(self):
        if self.last_stats is None:
            print("No generation statistics yet.")
            return

        stats = self.last_stats

        print("\n--- Performance ---")
        print(f"Prompt tokens: {stats['prompt_tokens']}")
        print(f"Prefill: {stats['prompt_tps']:.1f} tok/s")
        print(f"Generated: {stats['generation_tokens']} tokens")
        print(f"Decode: {stats['generation_tps']:.1f} tok/s")
        print(f"MLX peak memory: {stats['peak_memory_gb']:.2f} GB")
        print(f"Elapsed: {stats['elapsed_seconds']:.2f}s")
        print(f"Finish reason: {stats['finish_reason']}")


def main():
    session = ChatSession(ChatConfig())

    print("\nCommands: /help, /clear, /stats, /exit")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input == "/exit":
            break

        if user_input == "/help":
            print(" /clear - Clear conversation")
            print(" /stats - Last generation metrics")
            print(" /exit  - Quit")
            continue

        if user_input == "/clear":
            session.reset()
            print("Conversation cleared.")
            continue

        if user_input == "/stats":
            session.print_stats()
            continue

        if not user_input:
            continue

        try:
            session.generate(user_input)
        except ValueError as exc:
            print(f"Input error: {exc}")


if __name__ == "__main__":
    main()

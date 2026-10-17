
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3-8B-4bit"


def main():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, concise AI assistant.",
        }
    ]

    print("Model loaded! Type /exit to quit, /clear to reset.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input == "/exit":
            break

        if user_input == "/clear":
            messages = messages[:1]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        pending_messages = messages + [
            {"role": "user", "content": user_input}
        ]

        prompt = tokenizer.apply_chat_template(
            pending_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampler = make_sampler(
            temp=0.7,
            top_p=0.8,
            top_k=20,
        )

        print("\nAssistant: ", end="", flush=True)

        full_response = []

        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
        ):
            print(response.text, end="", flush=True)
            full_response.append(response.text)

        print()

        # Save only completed turns to conversation history.
        messages = pending_messages + [
            {
                "role": "assistant",
                "content": "".join(full_response),
            }
        ]


if __name__ == "__main__":
    main()

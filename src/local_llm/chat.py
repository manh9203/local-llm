from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine


def main():
    engine = InferenceEngine(ChatConfig())
    session = engine.create_session()

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
            print("\nAssistant: ", end="", flush=True)

            try:
                for chunk in session.stream(user_input):
                    print(chunk, end="", flush=True)
            except KeyboardInterrupt:
                print("\n[Generation interrupted]")
            finally:
                print()

            if session.last_stats is not None:
                session.print_stats()

        except ValueError as exc:
            print(f"Input error: {exc}")


if __name__ == "__main__":
    main()

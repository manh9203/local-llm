from local_llm.inference.engine import (
    ChatSession,
    InferenceEngine,
    common_prefix_length,
)


class SessionCacheManager:
    """Bounded cache of completed chat sessions."""

    def __init__(
        self,
        engine: InferenceEngine,
        max_sessions: int = 2,
        max_cached_tokens: int = 3000,
        min_prefix_tokens: int = 64,
    ):
        if max_sessions < 1 or max_cached_tokens < 1:
            raise ValueError("Cache limits must be positive.")

        self.engine = engine
        self.max_sessions = max_sessions
        self.max_cached_tokens = max_cached_tokens
        self.min_prefix_tokens = min_prefix_tokens

        # Least recently used -> most recently used.
        self.sessions: list[ChatSession] = []

    def acquire(self, messages: list[dict]) -> ChatSession:
        """Find a reusable session for the supplied conversation."""
        best_index = None
        best_prefix = 0

        if self.sessions:
            token_ids = self.sessions[0].tokenize_messages(messages)

            for i, session in enumerate(self.sessions):
                prefix = common_prefix_length(
                    session.cached_tokens,
                    token_ids,
                )

                if prefix > best_prefix:
                    best_index = i
                    best_prefix = prefix

        if (
            best_index is not None
            and best_prefix >= self.min_prefix_tokens
        ):
            session = self.sessions.pop(best_index)
        else:
            session = self.engine.create_session()

        # Inherit the conversation history, but not the last user message.
        session.messages = [m.copy() for m in messages[:-1]]

        return session

    def release(self, session: ChatSession) -> None:
        """Retain a successfully completed session if it fits."""
        if (
            not session.cache_enabled
            or session.prompt_cache is None
            or not session.cached_tokens
            or session.cache_size() != len(session.cached_tokens)
        ):
            session.clear_cache()
            return

        size = len(session.cached_tokens)

        if size > self.max_cached_tokens:
            session.clear_cache()
            return

        self.sessions.append(session)

        # Evict least recently used sessions.
        while len(self.sessions) > self.max_sessions:
            old = self.sessions.pop(0)
            old.clear_cache()

        # Enforce aggregate cached-token budget.
        while (
            sum(len(s.cached_tokens) for s in self.sessions)
            > self.max_cached_tokens
        ):
            old = self.sessions.pop(0)
            old.clear_cache()

    def clear(self) -> None:
        for session in self.sessions:
            session.clear_cache()

        self.sessions.clear()

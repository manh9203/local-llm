from types import SimpleNamespace

from local_llm.serving.session_cache import SessionCacheManager


class FakeSession:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cached_tokens = []
        self.prompt_cache = None
        self.cache_enabled = True
        self.messages = []

    def tokenize_messages(self, messages):
        return self.tokenizer(messages)

    def cache_size(self):
        return len(self.cached_tokens)

    def clear_cache(self):
        self.cached_tokens = []
        self.prompt_cache = None


class FakeEngine:
    def __init__(self):
        self.created = []

    def create_session(self):
        session = FakeSession(self.tokenize)
        self.created.append(session)
        return session

    @staticmethod
    def tokenize(messages):
        # Deterministic fake tokenization.
        return [
            ord(char)
            for message in messages
            for char in message["content"]
        ]


def messages(text):
    return [{"role": "user", "content": text}]


def populate(session, tokens):
    session.cached_tokens = tokens.copy()
    session.prompt_cache = SimpleNamespace()


def test_reuses_matching_prefix():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        min_prefix_tokens=3,
    )

    first = manager.acquire(messages("hello"))
    populate(first, engine.tokenize(messages("hello")))
    manager.release(first)

    second = manager.acquire(messages("hello world"))

    assert second is first
    assert second.cached_tokens == [
        ord(c) for c in "hello"
    ]


def test_different_prefix_creates_new_session():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        min_prefix_tokens=3,
    )

    first = manager.acquire(messages("hello"))
    populate(first, engine.tokenize(messages("hello")))
    manager.release(first)

    second = manager.acquire(messages("goodbye"))

    assert second is not first
    assert len(engine.created) == 2


def test_eviction_limits_sessions():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        max_sessions=1,
        min_prefix_tokens=3,
    )

    first = manager.acquire(messages("hello"))
    populate(first, [1, 2, 3])
    manager.release(first)

    second = manager.acquire(messages("goodbye"))
    populate(second, [4, 5, 6])
    manager.release(second)

    assert len(manager.sessions) == 1
    assert manager.sessions[0] is second
    assert first.prompt_cache is None


def test_token_budget_eviction():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        max_sessions=2,
        max_cached_tokens=6,
        min_prefix_tokens=3,
    )

    first = manager.acquire(messages("hello"))
    populate(first, [1, 2, 3, 4])
    manager.release(first)

    second = manager.acquire(messages("goodbye"))
    populate(second, [5, 6, 7, 8])
    manager.release(second)

    assert len(manager.sessions) == 1
    assert manager.sessions[0] is second
    assert first.prompt_cache is None


def test_clear_invalidates_all_caches():
    engine = FakeEngine()
    manager = SessionCacheManager(engine)

    session = manager.acquire(messages("hello"))
    populate(session, [1, 2, 3])
    manager.release(session)

    manager.clear()

    assert manager.sessions == []
    assert session.prompt_cache is None
    assert session.cached_tokens == []

from types import SimpleNamespace

from local_llm.serving.session_cache import SessionCacheManager


class FakeSession:
    def __init__(self, tokenizer, cache_strategy="trimmable"):
        self.tokenizer = tokenizer
        self.cache_strategy = cache_strategy
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

    def clone_cached_session(self):
        clone = FakeSession(
            self.tokenizer,
            cache_strategy=self.cache_strategy,
        )
        clone.cached_tokens = self.cached_tokens.copy()
        clone.prompt_cache = (
            SimpleNamespace()
            if self.prompt_cache is not None
            else None
        )
        clone.messages = [message.copy() for message in self.messages]
        return clone


class FakeEngine:
    def __init__(self, cache_strategy="trimmable"):
        self.created = []
        self.cache_strategy = cache_strategy

    def create_session(self):
        session = FakeSession(
            self.tokenize,
            cache_strategy=self.cache_strategy,
        )
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

    first = manager.acquire(engine.tokenize(messages("hello")))
    populate(first, engine.tokenize(messages("hello")))
    manager.release(first)

    second = manager.acquire(engine.tokenize(messages("hello world")))

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

    first = manager.acquire(engine.tokenize(messages("hello")))
    populate(first, engine.tokenize(messages("hello")))
    manager.release(first)

    second = manager.acquire(engine.tokenize(messages("goodbye")))

    assert second is not first
    assert len(engine.created) == 2


def test_eviction_limits_sessions():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        max_sessions=1,
        min_prefix_tokens=3,
    )

    first = manager.acquire(engine.tokenize(messages("hello")))
    populate(first, [1, 2, 3])
    manager.release(first)

    second = manager.acquire(engine.tokenize(messages("goodbye")))
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

    first = manager.acquire(engine.tokenize(messages("hello")))
    populate(first, [1, 2, 3, 4])
    manager.release(first)

    second = manager.acquire(engine.tokenize(messages("goodbye")))
    populate(second, [5, 6, 7, 8])
    manager.release(second)

    assert len(manager.sessions) == 1
    assert manager.sessions[0] is second
    assert first.prompt_cache is None


def test_clear_invalidates_all_caches():
    engine = FakeEngine()
    manager = SessionCacheManager(engine)

    session = manager.acquire(engine.tokenize(messages("hello")))
    populate(session, [1, 2, 3])
    manager.release(session)

    manager.clear()

    assert manager.sessions == []
    assert session.prompt_cache is None
    assert session.cached_tokens == []


def test_cache_selection_uses_prepared_tokens():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        min_prefix_tokens=3,
    )

    first = manager.acquire([1, 2, 3, 4, 5])
    populate(first, [1, 2, 3, 4, 5])
    manager.release(first)

    # This represents the effective prompt after truncation.
    # The original untruncated history is irrelevant.
    prepared_tokens = [1, 2, 3, 9, 10]

    second = manager.acquire(prepared_tokens)

    assert second is first
    assert second.cached_tokens == [1, 2, 3, 4, 5]


def test_retains_long_cache():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        max_sessions=1,
        max_cached_tokens=8192,
        min_prefix_tokens=64,
    )

    tokens = list(range(7000))

    session = manager.acquire(tokens)
    populate(session, tokens)

    manager.release(session)

    assert manager.sessions == [session]
    assert len(session.cached_tokens) == 7000

    # A subsequent prompt extends the same prefix.
    next_session = manager.acquire(tokens + [7000, 7001])

    assert next_session is session


def test_rejects_cache_above_budget():
    engine = FakeEngine()

    manager = SessionCacheManager(
        engine,
        max_sessions=1,
        max_cached_tokens=8192,
        min_prefix_tokens=64,
    )

    session = manager.acquire([1, 2, 3])
    populate(session, list(range(9000)))

    manager.release(session)

    assert manager.sessions == []
    assert session.prompt_cache is None
    assert session.cached_tokens == []


def test_append_only_preserves_checkpoint():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        max_sessions=5,
        min_prefix_tokens=3,
    )

    original = manager.acquire([1, 2, 3, 4])
    populate(original, [1, 2, 3, 4])
    manager.release(original)

    clone = manager.acquire([1, 2, 3, 4, 5])

    assert clone is not original
    assert clone.cached_tokens == original.cached_tokens
    assert clone.prompt_cache is not original.prompt_cache

    assert manager.sessions == [original]

    clone.cached_tokens.append(5)
    assert original.cached_tokens == [1, 2, 3, 4]


def test_append_only_reuses_checkpoint_for_two_forks():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        max_sessions=5,
        min_prefix_tokens=3,
    )

    checkpoint = manager.acquire([1, 2, 3])
    populate(checkpoint, [1, 2, 3])
    manager.release(checkpoint)

    branch_a = manager.acquire([1, 2, 3, 4])
    branch_b = manager.acquire([1, 2, 3, 5])

    assert branch_a is not branch_b
    assert branch_a is not checkpoint
    assert branch_b is not checkpoint

    branch_a.cached_tokens.append(4)
    branch_b.cached_tokens.append(5)

    assert checkpoint.cached_tokens == [1, 2, 3]
    assert checkpoint in manager.sessions


def test_append_only_rejects_partial_prefix():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        min_prefix_tokens=3,
    )

    original = manager.acquire([1, 2, 3, 4])
    populate(original, [1, 2, 3, 4])
    manager.release(original)

    selected = manager.acquire([1, 2, 9, 10])

    assert selected is not original
    assert original in manager.sessions


def test_append_only_moves_checkpoint_to_lru_front():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        max_sessions=5,
        min_prefix_tokens=3,
    )

    oldest = engine.create_session()
    populate(oldest, [10, 11, 12])

    checkpoint = engine.create_session()
    populate(checkpoint, [1, 2, 3])

    newest = engine.create_session()
    populate(newest, [20, 21, 22])

    manager.sessions = [oldest, checkpoint, newest]

    clone = manager.acquire([1, 2, 3, 4])

    assert clone is not checkpoint
    assert manager.sessions == [
        checkpoint,
        oldest,
        newest,
    ]


def test_append_only_checkpoint_evicted_first():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        max_sessions=2,
        min_prefix_tokens=3,
    )

    checkpoint = engine.create_session()
    populate(checkpoint, [1, 2, 3])

    other = engine.create_session()
    populate(other, [10, 11, 12])

    manager.sessions = [checkpoint, other]

    clone = manager.acquire([1, 2, 3, 4])
    populate(clone, [1, 2, 3, 4])

    manager.release(clone)

    assert checkpoint not in manager.sessions
    assert checkpoint.prompt_cache is None
    assert manager.sessions == [other, clone]


def test_append_only_clone_respects_token_budget():
    engine = FakeEngine(cache_strategy="append_only")
    manager = SessionCacheManager(
        engine,
        max_sessions=5,
        max_cached_tokens=7,
        min_prefix_tokens=3,
    )

    checkpoint = engine.create_session()
    populate(checkpoint, [1, 2, 3, 4])
    manager.release(checkpoint)

    clone = manager.acquire([1, 2, 3, 4, 5])
    populate(clone, [1, 2, 3, 4, 5])
    manager.release(clone)

    # Total size would be 9, exceeding the budget of 7.
    # The original checkpoint is evicted first.
    assert checkpoint not in manager.sessions
    assert checkpoint.prompt_cache is None
    assert manager.sessions == [clone]

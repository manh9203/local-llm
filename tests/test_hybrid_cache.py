from local_llm.inference import engine as engine_module
from local_llm.inference.engine import ChatSession


class FakeKVCache:
    def __init__(self, size=0):
        self.position = size

    def size(self):
        return self.position


class FakeArraysCache:
    def size(self):
        return 0


def make_session(monkeypatch, position, tokens):
    def new_cache(_model):
        return [
            FakeArraysCache(),
            FakeKVCache(),
        ]

    monkeypatch.setattr(
        engine_module,
        "make_prompt_cache",
        new_cache,
    )

    monkeypatch.setattr(
        engine_module,
        "KVCache",
        FakeKVCache,
    )

    session = ChatSession.__new__(ChatSession)
    session.model = object()
    session.cache_enabled = True
    session.cache_strategy = "append_only"
    session.prompt_cache = [
        FakeArraysCache(),
        FakeKVCache(position),
    ]
    session.cached_tokens = list(tokens)

    return session


def test_append_only_reuses_exact_prefix(monkeypatch):
    session = make_session(
        monkeypatch,
        position=3,
        tokens=[10, 20, 30],
    )

    suffix, reused = session.prepare_cache(
        [10, 20, 30, 40, 50]
    )

    assert suffix == [40, 50]
    assert reused == 3
    assert session.cached_tokens == [10, 20, 30]


def test_append_only_resets_on_divergence(monkeypatch):
    session = make_session(
        monkeypatch,
        position=3,
        tokens=[10, 20, 30],
    )

    suffix, reused = session.prepare_cache(
        [10, 99, 30, 40]
    )

    assert suffix == [10, 99, 30, 40]
    assert reused == 0
    assert session.cached_tokens == []
    assert session.cache_size() == 0


def test_append_only_resets_on_position_mismatch(monkeypatch):
    session = make_session(
        monkeypatch,
        position=4,
        tokens=[10, 20, 30],
    )

    suffix, reused = session.prepare_cache(
        [10, 20, 30, 40]
    )

    assert reused == 0
    assert suffix == [10, 20, 30, 40]


def test_append_only_records_processed_tokens(monkeypatch):
    session = make_session(
        monkeypatch,
        position=3,
        tokens=[10, 20, 30],
    )

    # Model processes 2 prompt tokens and 2 generated tokens.
    session.prompt_cache[1].position = 7

    session.update_cache(
        input_tokens=[40, 50],
        generated_tokens=[60, 70, 80],
    )

    assert session.cached_tokens == [
        10, 20, 30, 40, 50, 60, 70
    ]


def test_append_only_rejects_unaccounted_cache(monkeypatch):
    session = make_session(
        monkeypatch,
        position=3,
        tokens=[10, 20, 30],
    )

    session.prompt_cache[1].position = 10

    session.update_cache(
        input_tokens=[40],
        generated_tokens=[50],
    )

    assert session.prompt_cache is None
    assert session.cached_tokens == []

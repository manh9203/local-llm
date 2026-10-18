import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_llm.inference.context import prepare_context
from local_llm.serving.api import create_app

MODEL_ID = "mlx-community/Qwen3-8B-4bit"


class FakeSession:
    def __init__(self):
        self.messages = []
        self.last_stats = None
        self.cache_enabled = True
        self.prompt_cache = None
        self.cached_tokens = []

    def stream_prepared(self, prepared):
        self.last_context_stats = prepared.context_stats

        yield "Hello"
        yield " from MLX!"

        self.messages = [
            message.copy()
            for message in prepared.messages
        ] + [
            {
                "role": "assistant",
                "content": "Hello from MLX!",
            }
        ]

        self.last_stats = {
            "total_prompt_tokens": len(prepared.token_ids),
            "generation_tokens": 4,
            "finish_reason": "stop",
        }

    def prepare_prompt(self, user_input):
        return (
            self.messages + [
                {"role": "user", "content": user_input}
            ],
            [1, 2, 3],
        )

    def clear_cache(self):
        self.prompt_cache = None
        self.cached_tokens = []

    def cache_size(self):
        return len(self.cached_tokens)

    def tokenize_messages(self, messages):
        return [
            ord(c)
            for message in messages
            for c in message["content"]
        ]

    def prepare_messages(self, messages):
        return prepare_context(
            messages=messages,
            tokenize_messages=self.tokenize_messages,
            max_context_tokens=8192,
            max_new_tokens=512,
        )


class FakeEngine:
    def __init__(self):
        self.config = SimpleNamespace(
            model_id=MODEL_ID,
            system_prompt="You are helpful.",
        )
        self.sessions = []

    def create_session(self):
        session = FakeSession()
        self.sessions.append(session)
        return session


def test_health():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == MODEL_ID


def test_chat_completion():
    engine = FakeEngine()

    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["object"] == "chat.completion"
    assert (
        result["choices"][0]["message"]["content"]
        == "Hello from MLX!"
    )

    usage = result["usage"]

    assert usage["prompt_tokens"] == 21
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )

    # The preparation session does not modify conversation history.
    assert engine.sessions[0].messages == []

    # The generation session commits the completed conversation.
    assert engine.sessions[1].messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello from MLX!"},
    ]


def test_rejects_unknown_model():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-model",
                "messages": [
                    {"role": "user", "content": "Hi"}
                ],
            },
        )

    assert response.status_code == 404


def test_rejects_invalid_conversation():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "assistant", "content": "Hello"}
                ],
            },
        )

    assert response.status_code == 400


def parse_sse_events(body: str):
    """Parse JSON data events, excluding the DONE marker."""
    events = []

    for line in body.splitlines():
        if not line.startswith("data: "):
            continue

        data = line.removeprefix("data: ")

        if data != "[DONE]":
            events.append(json.loads(data))

    return events


def test_streaming_chat_completion():
    engine = FakeEngine()

    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = parse_sse_events(response.text)

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"

    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
    )

    assert content == "Hello from MLX!"

    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"][0]["delta"] == {}

    assert response.text.rstrip().endswith("data: [DONE]")

    # All events belong to the same completion.
    assert len({event["id"] for event in events}) == 1
    assert all(
        event["object"] == "chat.completion.chunk"
        for event in events
    )


def test_streaming_preserves_request_history():
    engine = FakeEngine()

    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up"},
                ],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.text.rstrip().endswith("data: [DONE]")

    # The first session is for prompt preparation.
    # The last session performs actual generation.
    session = engine.sessions[-1]

    assert session.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
        {"role": "assistant", "content": "Hello from MLX!"},
    ]

    assert session.last_context_stats.dropped_turns == 0

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_llm.serving.api import create_app

MODEL_ID = "mlx-community/Qwen3-8B-4bit"


class FakeSession:
    def __init__(self):
        self.messages = []
        self.last_stats = None

    def stream(self, user_input):
        assert user_input
        yield "Hello"
        yield " from MLX!"

        self.last_stats = {
            "total_prompt_tokens": 10,
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
    assert result["usage"]["total_tokens"] == 14

    # Prior messages exclude the latest user message.
    assert engine.sessions[0].messages == [
        {"role": "system", "content": "You are helpful."}
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

    session = engine.sessions[0]

    assert session.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]

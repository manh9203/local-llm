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


def test_rejects_streaming_for_now():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Hi"}
                ],
                "stream": True,
            },
        )

    assert response.status_code == 422

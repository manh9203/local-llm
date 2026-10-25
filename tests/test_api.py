import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_llm.conversation.compaction import CompactionConfig
from local_llm.conversation.manager import ConversationManager
from local_llm.conversation.service import CompactionService
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
        self.received_prompt = prepared

        yield "Hello"
        yield " from MLX!"

        self.messages = [m.copy() for m in prepared.messages] + [
            {"role": "assistant", "content": "Hello from MLX!"}
        ]

        self.last_context_stats = prepared.context_stats

        self.last_stats = {
            "total_prompt_tokens": len(prepared.token_ids),
            "reused_tokens": 0,
            "processed_prompt_tokens": len(prepared.token_ids),
            "prompt_tps": 100.0,
            "generation_tokens": 4,
            "generation_tps": 20.0,
            "peak_memory_gb": 1.0,
            "elapsed_seconds": 0.25,
            "first_token_seconds": 0.05,
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
            max_context_tokens=8192,
            max_new_tokens=512,
        )
        self.sessions = []

    def create_session(self):
        session = FakeSession()
        self.sessions.append(session)
        return session


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, messages, previous_summary, max_tokens):
        self.calls.append({
            "messages": messages,
            "previous_summary": previous_summary,
        })
        return "The user is building an MLX inference engine."


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


def test_api_compacts_before_generation():
    engine = FakeEngine()
    summarizer = FakeSummarizer()

    service = CompactionService(
        manager=ConversationManager(),
        summarizer=summarizer,
        config=CompactionConfig(
            trigger_utilization=0.01,
            recent_turns_to_keep=2,
            summary_max_tokens=1,
            min_turns_to_compact=2,
            min_new_tokens_to_compact=0,
            min_expected_savings_tokens=0,
            summary_overhead_tokens=0,
        ),
    )

    messages = []

    for i in range(5):
        messages.extend([
            {
                "role": "user",
                "content": (
                    f"Question {i}: "
                    + "I am developing a local LLM inference engine "
                    "with MLX, FastAPI, KV-cache reuse, and "
                    "conversation compaction. " * 5
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Answer {i}: "
                    + "We should optimize prefill latency, "
                    "decoding throughput, memory usage, "
                    "and prompt-cache retention. " * 5
                ),
            },
        ])

    messages.append({
        "role": "user",
        "content": "What should we do next?",
    })

    with TestClient(
        create_app(engine, compaction_service=service)
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": messages,
            },
        )

    assert response.status_code == 200
    assert len(summarizer.calls) == 1

    checkpoint = service.manager.find(
        [{"role": "system", "content": "You are helpful."}] + messages,
        MODEL_ID,
    )

    assert checkpoint is not None
    assert checkpoint.covered_count == 7

    generation_session = engine.sessions[-1]
    prepared = generation_session.received_prompt

    assert "MLX inference engine" in prepared.messages[0]["content"]

    assert len(prepared.messages) < len(messages) + 1

    assert prepared.messages[-1]["content"] == (
        "What should we do next?"
    )

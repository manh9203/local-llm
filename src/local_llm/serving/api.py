import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from local_llm.config import ChatConfig
from local_llm.inference.engine import InferenceEngine


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="forbid")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: Literal[False] = False

    model_config = ConfigDict(extra="forbid")


def validate_messages(messages: list[ChatMessage]) -> None:
    """Validate the conversation shape our engine supports."""
    if messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message must be from the user.",
        )

    start = 0

    if messages[0].role == "system":
        start = 1

    for index in range(start, len(messages)):
        expected = (
            "user" if (index - start) % 2 == 0
            else "assistant"
        )

        if messages[index].role != expected:
            raise HTTPException(
                status_code=400,
                detail="Expected alternating user/assistant messages.",
            )


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Create the API, optionally injecting an engine for tests."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            app.state.engine = InferenceEngine(ChatConfig())
        else:
            app.state.engine = engine

        yield

    app = FastAPI(
        title="Local MLX LLM",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Serialize access to shared model execution.
    inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": app.state.engine.config.model_id,
        }

    @app.get("/v1/models")
    def list_models():
        model_id = app.state.engine.config.model_id

        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        model_engine = app.state.engine

        if request.model != model_engine.config.model_id:
            raise HTTPException(
                status_code=404,
                detail="Requested model is not available.",
            )

        validate_messages(request.messages)

        # The request includes complete conversation history.
        messages = [
            message.model_dump()
            for message in request.messages
        ]

        session = model_engine.create_session()

        # Use the supplied system prompt, or our default.
        if messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": model_engine.config.system_prompt,
                },
            )

        # stream() appends the latest user message itself.
        session.messages = messages[:-1]
        latest_user_message = messages[-1]["content"]

        try:
            with inference_lock:
                answer = "".join(
                    session.stream(latest_user_message)
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        stats = session.last_stats

        if stats is None:
            raise HTTPException(
                status_code=500,
                detail="Inference did not complete.",
            )

        prompt_tokens = stats["total_prompt_tokens"]
        completion_tokens = stats["generation_tokens"]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_engine.config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": stats["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    prompt_tokens + completion_tokens
                ),
            },
        }

    return app


app = create_app()

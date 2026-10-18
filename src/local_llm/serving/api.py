import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, closing
from typing import Literal

import anyio
import anyio.lowlevel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
    stream: bool = False

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


_END = object()


def next_fragment(iterator):
    """Advance a synchronous MLX generator by one step."""
    return next(iterator, _END)


def sse_data(payload: dict) -> str:
    """Format one JSON object as an SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def completion_chunk(
    completion_id: str,
    created: int,
    model_id: str,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Create the API, optionally injecting an engine for tests."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = (
            engine if engine is not None
            else InferenceEngine(ChatConfig())
        )
        yield

    app = FastAPI(
        title="Local MLX LLM",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Serialize access to shared model execution.
    inference_lock = asyncio.Lock()

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
            "data": [{
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
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

        # Use the supplied system prompt, or our default.
        if messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system",
                "content": model_engine.config.system_prompt,
            })

        session = model_engine.create_session()

        # stream() appends the latest user message itself.
        session.messages = messages[:-1]
        latest_user_message = messages[-1]["content"]

        # Validate context before sending any HTTP response.
        try:
            session.prepare_prompt(latest_user_message)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_id = model_engine.config.model_id

        if not request.stream:
            def generate_all():
                with closing(session.stream(latest_user_message)) as stream:
                    return "".join(stream)

            async with inference_lock:
                answer = await anyio.to_thread.run_sync(
                    generate_all
                )

            stats = session.last_stats

            if stats is None:
                raise HTTPException(
                    status_code=500,
                    detail="Inference did not complete.",
                )

            prompt_tokens = stats["total_prompt_tokens"]
            completion_tokens = stats["generation_tokens"]

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": stats["finish_reason"],
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (
                        prompt_tokens + completion_tokens
                    ),
                },
            }

        async def events():
            async with inference_lock:
                stream = session.stream(latest_user_message)

                try:
                    yield sse_data(completion_chunk(
                        completion_id,
                        created,
                        model_id,
                        {"role": "assistant", "content": ""},
                    ))

                    while True:
                        fragment = await anyio.to_thread.run_sync(
                            next_fragment,
                            stream,
                        )

                        if fragment is _END:
                            break

                        if fragment:
                            yield sse_data(completion_chunk(
                                completion_id,
                                created,
                                model_id,
                                {"content": fragment},
                            ))

                        # Allow the event loop to observe cancellation.
                        await anyio.lowlevel.checkpoint()

                    stats = session.last_stats

                    if stats is None:
                        return

                    yield sse_data(completion_chunk(
                        completion_id,
                        created,
                        model_id,
                        {},
                        stats["finish_reason"],
                    ))

                    yield "data: [DONE]\n\n"

                finally:
                    # Close an unfinished MLX stream when the response is cancelled.
                    with anyio.CancelScope(shield=True):
                        await anyio.to_thread.run_sync(
                            stream.close
                        )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()

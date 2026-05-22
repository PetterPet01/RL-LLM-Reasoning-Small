"""
litert_lm_server.py — OpenAI-compatible server for LiteRT-LM
High-performance, maximum-throughput implementation.

Usage:
    pip install litert-lm-api-nightly fastapi uvicorn[standard] pydantic

    python litert_lm_server.py \
        --model path/to/model.litertlm \
        --host 0.0.0.0 \
        --port 8000 \
        --backend gpu \
        --max-concurrent-requests 32 \
        --conversation-pool-size 8 \
        --enable-mtp

OpenAI-compatible endpoints:
    GET  /v1/models
    POST /v1/chat/completions        (streaming + non-streaming)
    POST /v1/completions             (legacy, wraps chat)
    GET  /health
    GET  /metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("litert_lm_server")

# ---------------------------------------------------------------------------
# Pydantic schemas — OpenAI-compatible
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "litert-lm"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: int = 1
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[str | dict[str, Any]] = None

    @property
    def effective_max_tokens(self) -> Optional[int]:
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(BaseModel):
    model: str = "litert-lm"
    prompt: str | list[str]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _make_chat_chunk(
    request_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def _make_chat_response(
    request_id: str,
    model: str,
    content: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Conversation pool — reuse pre-warmed conversation objects
# ---------------------------------------------------------------------------

@dataclass
class _ConvSlot:
    """A reusable conversation slot managed by the pool."""
    conversation: Any          # litert_lm.Conversation
    in_use: bool = False
    created_at: float = field(default_factory=time.time)


class ConversationPool:
    """
    Thread-safe pool of pre-allocated LiteRT-LM conversations.

    Each request borrows a slot, uses it (with fresh system prompt injected
    each time via a new conversation created from the engine), and returns it.
    Because creating a new conversation is cheap compared to engine init,
    we keep a pool of *engine* objects and create conversations on demand,
    one per in-flight request, capped at pool_size.
    """

    def __init__(self, engine: Any, pool_size: int):
        self._engine = engine
        self._sem = asyncio.Semaphore(pool_size)
        self._pool_size = pool_size
        log.info("ConversationPool ready (size=%d)", pool_size)

    @asynccontextmanager
    async def acquire(self, system_prompt: Optional[str] = None):
        """Async context manager: yields a fresh conversation, releases slot on exit."""
        await self._sem.acquire()
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        # conversation creation is blocking — run in executor
        loop = asyncio.get_event_loop()
        conv_ctx = await loop.run_in_executor(
            None, lambda: self._engine.create_conversation(messages=messages or None)
        )
        conversation = conv_ctx.__enter__()
        try:
            yield conversation
        finally:
            try:
                conv_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._sem.release()


# ---------------------------------------------------------------------------
# Inference worker — runs blocking litert_lm calls off the event loop
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Wraps a litert_lm.Engine and exposes async streaming + non-streaming
    inference methods.  All blocking calls are dispatched to a dedicated
    ThreadPoolExecutor so the asyncio event loop is never blocked.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "cpu",
        enable_mtp: bool = False,
        conversation_pool_size: int = 8,
        cache_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self._pool: Optional[ConversationPool] = None
        self._engine_ctx = None
        self._engine = None

        # config
        self._backend_str = backend.upper()
        self._enable_mtp = enable_mtp
        self._pool_size = conversation_pool_size
        self._cache_dir = cache_dir

        # metrics
        self.requests_total: int = 0
        self.tokens_generated: int = 0
        self.start_time: float = time.time()

    def load(self):
        """Synchronous model load — call once at startup."""
        import litert_lm  # noqa: import inside to keep module importable without litert_lm

        backend_map = {
            "CPU": litert_lm.Backend.CPU,
            "GPU": litert_lm.Backend.GPU,
        }
        backend = backend_map.get(self._backend_str, litert_lm.Backend.CPU)

        # Suppress verbose init logs
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        kwargs: dict[str, Any] = dict(backend=backend)
        if self._enable_mtp:
            kwargs["enable_speculative_decoding"] = True
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir

        log.info("Loading model: %s  backend=%s  mtp=%s", self.model_path, self._backend_str, self._enable_mtp)
        t0 = time.time()
        self._engine_ctx = litert_lm.Engine(self.model_path, **kwargs)
        self._engine = self._engine_ctx.__enter__()
        log.info("Model loaded in %.1fs", time.time() - t0)

        self._pool = ConversationPool(self._engine, self._pool_size)

    def unload(self):
        if self._engine_ctx is not None:
            try:
                self._engine_ctx.__exit__(None, None, None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(req: ChatCompletionRequest) -> tuple[list[dict], Optional[str]]:
        """
        Split system message out (for conversation init) and build the
        remaining turns as a list.  Returns (turns, system_prompt).
        """
        system_prompt: Optional[str] = None
        turns: list[dict] = []

        for msg in req.messages:
            if msg.role == "system":
                # Collect all system messages into one
                if isinstance(msg.content, str):
                    system_prompt = (system_prompt or "") + msg.content
                elif isinstance(msg.content, list):
                    text_parts = [p["text"] for p in msg.content if p.get("type") == "text"]
                    system_prompt = (system_prompt or "") + "\n".join(text_parts)
            else:
                # Normalise content to litert_lm format
                if isinstance(msg.content, str):
                    content = [{"type": "text", "text": msg.content}]
                else:
                    content = msg.content  # already structured
                turns.append({"role": msg.role, "content": content})

        return turns, system_prompt

    # ------------------------------------------------------------------
    # Non-streaming inference
    # ------------------------------------------------------------------

    async def chat(self, req: ChatCompletionRequest) -> dict[str, Any]:
        turns, system_prompt = self._build_messages(req)

        async with self._pool.acquire(system_prompt) as conv:
            # Replay prior turns (all but the last)
            loop = asyncio.get_event_loop()
            for turn in turns[:-1]:
                await loop.run_in_executor(None, conv.send_message, turn)

            # Final turn — collect full response
            last_turn = turns[-1] if turns else {"role": "user", "content": [{"type": "text", "text": ""}]}

            def _sync_generate():
                return conv.send_message(last_turn)

            response = await loop.run_in_executor(None, _sync_generate)
            content = response["content"][0]["text"] if response.get("content") else ""

        self.requests_total += 1
        completion_tokens = len(content.split())  # rough estimate
        self.tokens_generated += completion_tokens

        return _make_chat_response(
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            model=self.model_name,
            content=content,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    async def chat_stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        turns, system_prompt = self._build_messages(req)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        # Yield the role delta first
        yield _make_chat_chunk(request_id, self.model_name, {"role": "assistant", "content": ""})

        async with self._pool.acquire(system_prompt) as conv:
            loop = asyncio.get_event_loop()

            # Replay prior turns
            for turn in turns[:-1]:
                await loop.run_in_executor(None, conv.send_message, turn)

            last_turn = turns[-1] if turns else {"role": "user", "content": [{"type": "text", "text": ""}]}

            # We run send_message_async (which is a blocking iterator) in a
            # background thread and feed chunks through an asyncio.Queue so
            # the event loop stays responsive.
            chunk_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
            completion_tokens = 0

            def _produce():
                """Runs in executor; pushes text chunks onto the queue."""
                try:
                    stream = conv.send_message_async(last_turn)
                    for chunk in stream:
                        for item in chunk.get("content", []):
                            if item.get("type") == "text" and item["text"]:
                                # Non-blocking put with backpressure via blocking put
                                asyncio.run_coroutine_threadsafe(
                                    chunk_queue.put(item["text"]), loop
                                ).result(timeout=30)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(f"__ERROR__:{exc}"), loop
                    ).result(timeout=5)
                finally:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(None), loop  # sentinel
                    ).result(timeout=5)

            # Start producer in thread
            future = loop.run_in_executor(None, _produce)

            # Consume chunks
            while True:
                text = await chunk_queue.get()
                if text is None:
                    break
                if isinstance(text, str) and text.startswith("__ERROR__:"):
                    error_msg = text[len("__ERROR__:"):]
                    log.error("Inference error: %s", error_msg)
                    break
                completion_tokens += 1
                self.tokens_generated += 1
                yield _make_chat_chunk(request_id, self.model_name, {"content": text})

            await future  # ensure thread completed

        self.requests_total += 1

        # Final chunk with finish_reason
        include_usage = (req.stream_options or {}).get("include_usage", False)
        usage = {"prompt_tokens": 0, "completion_tokens": completion_tokens,
                 "total_tokens": completion_tokens} if include_usage else None
        yield _make_chat_chunk(request_id, self.model_name, {}, finish_reason="stop", usage=usage)
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Global engine instance (populated at startup)
# ---------------------------------------------------------------------------

_engine: Optional[InferenceEngine] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # startup
    global _engine
    assert _engine is not None, "Engine must be set before startup"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _engine.load)
    log.info("Server ready.")
    yield
    # shutdown
    if _engine:
        _engine.unload()
    log.info("Server shut down.")


app = FastAPI(
    title="LiteRT-LM OpenAI-compatible API",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _engine.model_name if _engine else None}


@app.get("/metrics")
async def metrics():
    if not _engine:
        raise HTTPException(503, "Engine not loaded")
    uptime = time.time() - _engine.start_time
    return {
        "uptime_seconds": round(uptime, 1),
        "requests_total": _engine.requests_total,
        "tokens_generated": _engine.tokens_generated,
        "tokens_per_second": round(_engine.tokens_generated / max(uptime, 1), 1),
    }


@app.get("/v1/models")
async def list_models():
    name = _engine.model_name if _engine else "litert-lm"
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(_engine.start_time) if _engine else 0,
                "owned_by": "litert-lm",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    if not _engine or not _engine._pool:
        raise HTTPException(503, "Engine not ready")

    if not req.messages:
        raise HTTPException(400, "messages must not be empty")

    if req.stream:
        async def _sse():
            async for chunk in _engine.chat_stream(req):
                yield chunk

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await _engine.chat(req)
        return JSONResponse(result)


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """Legacy completions endpoint — wraps chat completions."""
    if not _engine or not _engine._pool:
        raise HTTPException(503, "Engine not ready")

    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    prompt_text = prompts[0]  # handle single prompt; multi-prompt batching not supported

    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=prompt_text)],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream,
        stop=req.stop,
    )

    if req.stream:
        request_id = f"cmpl-{uuid.uuid4().hex}"

        async def _sse():
            async for chunk_str in _engine.chat_stream(chat_req):
                # Re-map object type for legacy completions
                if chunk_str.startswith("data: {"):
                    obj = json.loads(chunk_str[6:])
                    obj["object"] = "text_completion"
                    choice = obj["choices"][0]
                    # Move delta.content → text
                    text = choice.get("delta", {}).get("content", "")
                    obj["choices"][0] = {
                        "index": 0,
                        "text": text,
                        "finish_reason": choice.get("finish_reason"),
                    }
                    yield f"data: {json.dumps(obj)}\n\n"
                else:
                    yield chunk_str

        return StreamingResponse(_sse(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result = await _engine.chat(chat_req)
    # Re-map to completions response shape
    text = result["choices"][0]["message"]["content"]
    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": result["created"],
        "model": result["model"],
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": result["usage"],
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="LiteRT-LM OpenAI-compatible server")
    p.add_argument("--model", required=True, help="Path to .litertlm model file")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                   help="Inference backend (default: cpu)")
    p.add_argument("--enable-mtp", action="store_true",
                   help="Enable Multi-Token Prediction / speculative decoding (GPU only)")
    p.add_argument("--conversation-pool-size", "--max-concurrent-requests",
                   type=int, default=8, dest="conversation_pool_size",
                   help="Max concurrent in-flight requests (default: 8)")
    p.add_argument("--cache-dir", default=None,
                   help="Directory for caching compiled artifacts")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of uvicorn workers (default: 1; use 1 for GPU)")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    return p.parse_args()


def main():
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    global _engine
    _engine = InferenceEngine(
        model_path=args.model,
        backend=args.backend,
        enable_mtp=args.enable_mtp,
        conversation_pool_size=args.conversation_pool_size,
        cache_dir=args.cache_dir,
    )

    # Pass `app` object directly (not an import string).
    # Using an import string causes uvicorn to re-import the module in a
    # fresh process where _engine is None, breaking the lifespan assert.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # Performance tuning
        loop="uvloop",          # use uvloop if available; falls back to asyncio
        http="httptools",       # faster HTTP parser
        timeout_keep_alive=30,
        limit_concurrency=args.conversation_pool_size * 2,
        backlog=512,
    )


if __name__ == "__main__":
    main()
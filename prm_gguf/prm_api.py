"""
PRM (Process Reward Model) API Server
======================================
FastAPI service exposing math-shepherd-mistral-7b-prm for step-to-reward scoring.

Architecture
------------
- N worker processes, each owns one Llama instance (avoids GIL, real parallelism).
- Requests arrive at the HTTP layer, get pushed onto an asyncio queue.
- A batch-collector picks up to BATCH_SIZE requests every BATCH_TIMEOUT seconds
  and dispatches them to the worker pool (ProcessPoolExecutor).
- Each worker scores all items in its sub-batch sequentially (one GPU context per
  process), so GPU memory stays predictable.
- Results are returned via per-request asyncio.Future objects.

Endpoints
---------
POST /score        – score a single (question, output) pair
POST /score/batch  – score many pairs in one call
GET  /health       – liveness + queue depth
GET  /metrics      – throughput counters
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, Future as CFFuture
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.environ.get(
    "PRM_MODEL_PATH",
    "/AIClub_NAS/core_baotg/quanhm/models/math-shepherd-mistral-7b-prm.Q4_K_M.gguf",
)
N_GPU_LAYERS: int = int(os.environ.get("PRM_N_GPU_LAYERS", "32"))
N_CTX: int = int(os.environ.get("PRM_N_CTX", "8192"))
NUM_WORKERS: int = int(os.environ.get("PRM_NUM_WORKERS", "1"))
MAX_BATCH_SIZE: int = int(os.environ.get("PRM_MAX_BATCH_SIZE", "4"))
BATCH_TIMEOUT: float = float(os.environ.get("PRM_BATCH_TIMEOUT", "0.05"))  # seconds
HOST: str = os.environ.get("PRM_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PRM_PORT", "8787"))
LOG_LEVEL: str = os.environ.get("PRM_LOG_LEVEL", "info")

GOOD_TOKEN = "+"
BAD_TOKEN = "-"
STEP_TAG = "ки"

EXPECTED_CANDIDATE_TOKENS = [648, 387]
EXPECTED_STEP_TAG_ID = 12902

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("prm_api")

# ---------------------------------------------------------------------------
# Pure functions that run inside worker processes
# (must be top-level so they are picklable)
# ---------------------------------------------------------------------------

def _softmax2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _worker_init(model_path: str, n_gpu_layers: int, n_ctx: int) -> None:
    """Called once in each worker process to load the model into a global."""
    global _LLM, _CANDIDATE_TOKENS, _STEP_TAG_ID, _WORKER_N_CTX  # noqa: PLW0603

    from llama_cpp import Llama  # imported inside worker so parent process stays light

    log.info("Worker PID %d loading model …", os.getpid())
    _LLM = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )

    candidate_tokens = _LLM.tokenize(
        f"{GOOD_TOKEN} {BAD_TOKEN}".encode("utf-8"), add_bos=True
    )[1:]
    step_tag_id = _LLM.tokenize(STEP_TAG.encode("utf-8"), add_bos=False)[-1]

    _CANDIDATE_TOKENS = candidate_tokens
    _STEP_TAG_ID = step_tag_id
    _WORKER_N_CTX = n_ctx
    log.info(
        "Worker PID %d ready. candidate_tokens=%s step_tag_id=%d",
        os.getpid(),
        candidate_tokens,
        step_tag_id,
    )


def _score_one(question: str, output: str) -> List[float]:
    """Score a single (question, output) pair inside a worker process."""
    llm = _LLM  # noqa: F821  (set by _worker_init)
    candidate_tokens = _CANDIDATE_TOKENS  # noqa: F821
    step_tag_id = _STEP_TAG_ID  # noqa: F821
    worker_n_ctx = _WORKER_N_CTX  # noqa: F821

    text = f"{question} {output}"
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)

    # Truncate tokens to fit inside the KV cache context window.
    # Exceeding n_ctx causes `llama_decode returned 1` (failed to find a memory slot).
    if len(tokens) > worker_n_ctx:
        log.warning("Truncating sequence of length %d to N_CTX=%d", len(tokens), worker_n_ctx)
        tokens = tokens[:worker_n_ctx]

    llm.reset()
    llm.eval(tokens)

    scores_arr = np.asarray(llm.scores)
    if len(scores_arr) < len(tokens):
        raise RuntimeError(
            f"llm.scores shorter than tokens: {len(scores_arr)} < {len(tokens)}"
        )

    step_scores: List[float] = []
    for i, tok in enumerate(tokens):
        if tok == step_tag_id:
            logits_pm = scores_arr[i, candidate_tokens]
            prob_good = float(_softmax2(logits_pm)[0])
            step_scores.append(prob_good)

    return step_scores


def _score_batch_worker(
    pairs: List[tuple[str, str]]
) -> List[List[float]]:
    """Score a list of (question, output) pairs sequentially in one worker."""
    return [_score_one(q, o) for q, o in pairs]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    question: str = Field(..., description="The math problem or question text.")
    output: str = Field(
        ...,
        description=(
            "The step-by-step solution, with each step ending in the Cyrillic "
            "character 'ки' as the step separator."
        ),
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional caller-provided request ID (UUID generated if omitted).",
    )


class StepScore(BaseModel):
    step_index: int
    score: float


class ScoreResponse(BaseModel):
    request_id: str
    step_scores: List[float] = Field(
        ..., description="P(correct) for each step tag found in `output`."
    )
    step_details: List[StepScore]
    final_score: Optional[float] = Field(
        None, description="Score of the last step (proxy for overall quality)."
    )
    min_score: Optional[float]
    num_steps: int
    latency_ms: float


class BatchScoreRequest(BaseModel):
    items: List[ScoreRequest] = Field(..., max_length=256)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total_latency_ms: float
    num_items: int


class HealthResponse(BaseModel):
    status: str
    queue_depth: int
    num_workers: int
    tokenizer_ok: bool
    model_path: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_batches: int
    total_errors: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Internal queue item
# ---------------------------------------------------------------------------

@dataclass
class _QueueItem:
    request_id: str
    question: str
    output: str
    future: asyncio.Future  # resolved with List[float] or an Exception
    enqueued_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class _AppState:
    def __init__(self) -> None:
        self.executor: Optional[ProcessPoolExecutor] = None
        self.queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self.batcher_task: Optional[asyncio.Task] = None
        self.start_time: float = time.monotonic()

        # metrics
        self.total_requests: int = 0
        self.total_batches: int = 0
        self.total_errors: int = 0

        # tokenizer sanity (checked after first worker initialises)
        self.tokenizer_ok: bool = True


_state = _AppState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PRM Scoring API",
    description="Step-to-reward scoring using math-shepherd-mistral-7b-prm.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    log.info("Starting PRM API server …")
    log.info("  model     : %s", MODEL_PATH)
    log.info("  workers   : %d", NUM_WORKERS)
    log.info("  gpu_layers: %d", N_GPU_LAYERS)
    log.info("  ctx       : %d", N_CTX)
    log.info("  batch_size: %d", MAX_BATCH_SIZE)
    log.info("  timeout   : %.3fs", BATCH_TIMEOUT)

    _state.executor = ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(MODEL_PATH, N_GPU_LAYERS, N_CTX),
    )

    # Warm up workers so the first real request is fast
    loop = asyncio.get_event_loop()
    warm_futs = [
        loop.run_in_executor(_state.executor, _score_batch_worker, [("warmup", "ки")])
        for _ in range(NUM_WORKERS)
    ]
    await asyncio.gather(*warm_futs, return_exceptions=True)
    log.info("All workers warmed up.")

    _state.batcher_task = asyncio.create_task(_batcher_loop())
    log.info("Batcher loop started.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    log.info("Shutting down …")
    if _state.batcher_task:
        _state.batcher_task.cancel()
        try:
            await _state.batcher_task
        except asyncio.CancelledError:
            pass
    if _state.executor:
        _state.executor.shutdown(wait=False)
    log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Batcher loop
# ---------------------------------------------------------------------------

async def _batcher_loop() -> None:
    """
    Continuously drains the queue in batches and dispatches to the executor.

    Strategy
    --------
    - Wait up to BATCH_TIMEOUT for the first item.
    - Once the first item arrives, greedily collect up to MAX_BATCH_SIZE items
      without further waiting (drain what's already there).
    - Split the collected batch evenly across NUM_WORKERS, dispatch each
      sub-batch to a separate worker via run_in_executor.
    - Resolve each QueueItem's future when its result arrives.
    """
    loop = asyncio.get_event_loop()

    while True:
        # ---- collect a batch ------------------------------------------------
        batch: List[_QueueItem] = []

        try:
            # block until at least one item
            first = await asyncio.wait_for(_state.queue.get(), timeout=BATCH_TIMEOUT)
            batch.append(first)
        except asyncio.TimeoutError:
            continue

        # greedily drain
        while len(batch) < MAX_BATCH_SIZE:
            try:
                item = _state.queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        _state.total_batches += 1
        log.debug("Dispatching batch of %d items", len(batch))

        # ---- split across workers -------------------------------------------
        chunk_size = max(1, (len(batch) + NUM_WORKERS - 1) // NUM_WORKERS)
        chunks: List[List[_QueueItem]] = [
            batch[i : i + chunk_size] for i in range(0, len(batch), chunk_size)
        ]

        # Build (pairs, item_list) per chunk
        dispatch = [
            ([(it.question, it.output) for it in chunk], chunk)
            for chunk in chunks
        ]

        # ---- submit to executor --------------------------------------------
        async def _run_chunk(pairs, items):
            try:
                results: List[List[float]] = await loop.run_in_executor(
                    _state.executor, _score_batch_worker, pairs
                )
                for item, result in zip(items, results):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                _state.total_errors += len(items)
                log.exception("Worker error: %s", exc)
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)

        await asyncio.gather(*[_run_chunk(p, it) for p, it in dispatch])


# ---------------------------------------------------------------------------
# Helper: enqueue a single scoring job and await the result
# ---------------------------------------------------------------------------

async def _enqueue_and_wait(
    request_id: str, question: str, output: str
) -> List[float]:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    item = _QueueItem(
        request_id=request_id,
        question=question,
        output=output,
        future=fut,
    )
    await _state.queue.put(item)
    return await fut


def _build_response(
    request_id: str,
    step_scores: List[float],
    latency_ms: float,
) -> ScoreResponse:
    details = [StepScore(step_index=i, score=s) for i, s in enumerate(step_scores)]
    return ScoreResponse(
        request_id=request_id,
        step_scores=step_scores,
        step_details=details,
        final_score=step_scores[-1] if step_scores else None,
        min_score=min(step_scores) if step_scores else None,
        num_steps=len(step_scores),
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoreResponse, summary="Score a single solution")
async def score(req: ScoreRequest) -> ScoreResponse:
    """
    Score a single step-by-step solution.

    Each step in `output` must end with the Cyrillic step-tag character **ки**.
    Returns a P(correct) score per step, plus final/min aggregates.
    """
    request_id = req.request_id or str(uuid.uuid4())
    _state.total_requests += 1
    t0 = time.monotonic()

    try:
        step_scores = await _enqueue_and_wait(request_id, req.question, req.output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.monotonic() - t0) * 1000
    return _build_response(request_id, step_scores, latency_ms)


@app.post(
    "/score/batch",
    response_model=BatchScoreResponse,
    summary="Score multiple solutions in one call",
)
async def score_batch(req: BatchScoreRequest) -> BatchScoreResponse:
    """
    Score up to 256 (question, output) pairs in a single request.

    All items are pushed onto the shared queue and processed with maximum
    parallelism across available workers.
    """
    t0 = time.monotonic()
    _state.total_requests += len(req.items)

    tasks = [
        _enqueue_and_wait(
            item.request_id or str(uuid.uuid4()),
            item.question,
            item.output,
        )
        for item in req.items
    ]

    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    responses: List[ScoreResponse] = []
    for orig_item, raw in zip(req.items, results_raw):
        rid = orig_item.request_id or "unknown"
        if isinstance(raw, Exception):
            _state.total_errors += 1
            # Return empty scores rather than aborting the whole batch
            responses.append(
                ScoreResponse(
                    request_id=rid,
                    step_scores=[],
                    step_details=[],
                    final_score=None,
                    min_score=None,
                    num_steps=0,
                    latency_ms=0.0,
                )
            )
        else:
            latency_ms = (time.monotonic() - t0) * 1000
            responses.append(_build_response(rid, raw, latency_ms))

    total_latency_ms = (time.monotonic() - t0) * 1000
    return BatchScoreResponse(
        results=responses,
        total_latency_ms=round(total_latency_ms, 2),
        num_items=len(responses),
    )


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        queue_depth=_state.queue.qsize(),
        num_workers=NUM_WORKERS,
        tokenizer_ok=_state.tokenizer_ok,
        model_path=MODEL_PATH,
    )


@app.get("/metrics", response_model=MetricsResponse, summary="Throughput counters")
async def metrics() -> MetricsResponse:
    return MetricsResponse(
        total_requests=_state.total_requests,
        total_batches=_state.total_batches,
        total_errors=_state.total_errors,
        uptime_seconds=round(time.monotonic() - _state.start_time, 1),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "prm_api:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        workers=1,  # process-level workers handled by our executor; keep uvicorn single
    )
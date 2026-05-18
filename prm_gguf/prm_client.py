"""
PRM API – async Python client
==============================
Drop-in replacement for calling the scoring logic directly.
Uses httpx for async HTTP, with connection pooling and retry logic.

Usage
-----
    from prm_client import PRMClient

    async with PRMClient("http://localhost:8787") as client:
        # single
        resp = await client.score("What is 2+2?", "Step 1: 2+2=4. ки")
        print(resp.final_score)

        # batch – all requests in-flight concurrently
        responses = await client.score_batch([
            ("question 1", "solution 1"),
            ("question 2", "solution 2"),
        ])
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx


@dataclass
class StepScore:
    step_index: int
    score: float


@dataclass
class ScoreResult:
    request_id: str
    step_scores: List[float]
    step_details: List[StepScore]
    final_score: Optional[float]
    min_score: Optional[float]
    num_steps: int
    latency_ms: float

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreResult":
        return cls(
            request_id=d["request_id"],
            step_scores=d["step_scores"],
            step_details=[StepScore(**s) for s in d["step_details"]],
            final_score=d.get("final_score"),
            min_score=d.get("min_score"),
            num_steps=d["num_steps"],
            latency_ms=d["latency_ms"],
        )


class PRMClient:
    """Async client for the PRM Scoring API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        timeout: float = 120.0,
        max_connections: int = 64,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def __aenter__(self) -> "PRMClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        question: str,
        output: str,
        request_id: Optional[str] = None,
    ) -> ScoreResult:
        """Score a single (question, output) pair."""
        payload = {"question": question, "output": output}
        if request_id:
            payload["request_id"] = request_id

        resp = await self._client.post("/score", json=payload)
        resp.raise_for_status()
        return ScoreResult.from_dict(resp.json())

    async def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        request_ids: Optional[List[Optional[str]]] = None,
    ) -> List[ScoreResult]:
        """
        Score multiple (question, output) pairs in one HTTP call.

        The server will fan them out across workers automatically.
        """
        items = []
        for i, (q, o) in enumerate(pairs):
            item = {"question": q, "output": o}
            if request_ids and request_ids[i]:
                item["request_id"] = request_ids[i]
            items.append(item)

        resp = await self._client.post("/score/batch", json={"items": items})
        resp.raise_for_status()
        data = resp.json()
        return [ScoreResult.from_dict(r) for r in data["results"]]

    async def health(self) -> dict:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def metrics(self) -> dict:
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience: best-of-N re-ranking
    # ------------------------------------------------------------------

    async def rerank(
        self,
        question: str,
        candidates: List[str],
        aggregation: str = "final",
    ) -> Tuple[int, List[float]]:
        """
        Score N candidate solutions for the same question and return
        the index + aggregate scores sorted best-first.

        aggregation: 'final' | 'min' | 'mean'
        """
        pairs = [(question, c) for c in candidates]
        results = await self.score_batch(pairs)

        def _agg(r: ScoreResult) -> float:
            if not r.step_scores:
                return 0.0
            if aggregation == "final":
                return r.final_score or 0.0
            elif aggregation == "min":
                return r.min_score or 0.0
            else:  # mean
                return sum(r.step_scores) / len(r.step_scores)

        agg_scores = [_agg(r) for r in results]
        best_idx = int(max(range(len(agg_scores)), key=lambda i: agg_scores[i]))
        return best_idx, agg_scores


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

async def _smoke_test() -> None:
    async with PRMClient() as client:
        print("Health:", await client.health())

        result = await client.score(
            question=(
                "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and "
                "bakes 4 for her friends. She sells the rest at $2 each. How much?"
            ),
            output=(
                "Step 1: 16 eggs per day. ки\n"
                "Step 2: 16 - 3 = 13 eggs left. ки\n"
                "Step 3: 13 - 4 = 9 eggs left. ки\n"
                "Step 4: 9 * $2 = $18. The answer is: 18 ки"
            ),
        )
        print("Single score result:")
        print(f"  steps      : {result.num_steps}")
        print(f"  step_scores: {result.step_scores}")
        print(f"  final_score: {result.final_score}")
        print(f"  latency_ms : {result.latency_ms}")

        print("\nMetrics:", await client.metrics())


if __name__ == "__main__":
    asyncio.run(_smoke_test())
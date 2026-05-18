"""
Benchmark the PRM API throughput.

Examples
--------
    python3 bench_prm_api.py --requests 100 --concurrency 8
    python3 bench_prm_api.py --endpoint batch --requests 256 --batch-size 16 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import httpx


DEFAULT_BASE_URL = "https://prm.24102006.xyz"


SAMPLE_ITEMS = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
            "morning and bakes muffins for her friends every day with four. She "
            "sells the remainder at the farmers' market daily for $2 per fresh "
            "duck egg. How much in dollars does she make every day?"
        ),
        "output": (
            "Step 1: Janet's ducks lay 16 eggs per day. ки\n"
            "Step 2: She eats 3 eggs, so 16 - 3 = 13 eggs remain. ки\n"
            "Step 3: She bakes with 4 eggs, so 13 - 4 = 9 eggs remain. ки\n"
            "Step 4: She sells 9 eggs at $2 each, so 9 * 2 = 18. The answer is: 18 ки"
        ),
    },
    {
        "question": (
            "Tom has 5 apples. He buys 7 more apples and then gives 3 apples to "
            "his friend. How many apples does Tom have left?"
        ),
        "output": (
            "Step 1: Tom starts with 5 apples. ки\n"
            "Step 2: He buys 7 more, so 5 + 7 = 12 apples. ки\n"
            "Step 3: He gives away 3, so 12 - 3 = 9 apples. The answer is: 9 ки"
        ),
    },
    {
        "question": (
            "A car travels 180 miles in 3 hours at a constant speed. What is the "
            "car's speed in miles per hour?"
        ),
        "output": (
            "Step 1: Speed equals distance divided by time. ки\n"
            "Step 2: The distance is 180 miles and the time is 3 hours. ки\n"
            "Step 3: 180 / 3 = 60, so the speed is 60 miles per hour. The answer is: 60 ки"
        ),
    },
    {
        "question": "A rectangle has length 12 cm and width 5 cm. What is its area?",
        "output": (
            "Step 1: The area of a rectangle is length times width. ки\n"
            "Step 2: The length is 12 cm and the width is 5 cm. ки\n"
            "Step 3: 12 * 5 = 60, so the area is 60 square centimeters. The answer is: 60 ки"
        ),
    },
]


@dataclass
class RequestResult:
    ok: bool
    num_items: int
    latency_ms: float
    error: str | None = None


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def make_items(num_items: int) -> List[dict[str, str]]:
    samples: Iterable[dict[str, str]] = itertools.islice(
        itertools.cycle(SAMPLE_ITEMS),
        num_items,
    )
    items = []
    for item in samples:
        items.append(
            {
                "question": item["question"],
                "output": item["output"],
                "request_id": f"bench-{uuid.uuid4()}",
            }
        )
    return items


def chunked(items: Sequence[dict[str, str]], chunk_size: int) -> List[List[dict[str, str]]]:
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


async def post_score(client: httpx.AsyncClient, item: dict[str, str]) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.post("/score", json=item)
        response.raise_for_status()
        data = response.json()
        if data.get("num_steps", 0) == 0:
            raise RuntimeError("response contained zero scored steps")
        return RequestResult(
            ok=True,
            num_items=1,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            ok=False,
            num_items=1,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )


async def post_batch(
    client: httpx.AsyncClient,
    items: Sequence[dict[str, str]],
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.post("/score/batch", json={"items": list(items)})
        response.raise_for_status()
        data = response.json()
        results: list[dict[str, Any]] = data["results"]
        failed = sum(1 for result in results if result.get("num_steps", 0) == 0)
        if failed:
            raise RuntimeError(f"{failed}/{len(results)} batch items returned zero scored steps")
        return RequestResult(
            ok=True,
            num_items=len(items),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            ok=False,
            num_items=len(items),
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )


async def run_load(
    client: httpx.AsyncClient,
    endpoint: str,
    work_units: Sequence[dict[str, str] | List[dict[str, str]]],
    concurrency: int,
) -> List[RequestResult]:
    queue: asyncio.Queue[dict[str, str] | List[dict[str, str]]] = asyncio.Queue()
    for unit in work_units:
        queue.put_nowait(unit)

    results: list[RequestResult] = []
    results_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if endpoint == "score":
                result = await post_score(client, unit)  # type: ignore[arg-type]
            else:
                result = await post_batch(client, unit)  # type: ignore[arg-type]

            async with results_lock:
                results.append(result)
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def fetch_json(client: httpx.AsyncClient, path: str) -> dict[str, Any] | None:
    try:
        response = await client.get(path)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def print_report(
    args: argparse.Namespace,
    results: Sequence[RequestResult],
    elapsed_s: float,
    health: dict[str, Any] | None,
    metrics_before: dict[str, Any] | None,
    metrics_after: dict[str, Any] | None,
) -> None:
    ok_results = [result for result in results if result.ok]
    failed_results = [result for result in results if not result.ok]
    latencies = [result.latency_ms for result in ok_results]
    completed_items = sum(result.num_items for result in ok_results)
    attempted_items = sum(result.num_items for result in results)

    print("=" * 80)
    print("PRM API throughput benchmark")
    print("=" * 80)
    print(f"base_url       : {args.base_url}")
    print(f"endpoint       : /{args.endpoint if args.endpoint == 'score' else 'score/batch'}")
    print(f"attempted_items: {attempted_items}")
    print(f"completed_items: {completed_items}")
    print(f"http_requests  : {len(results)}")
    print(f"concurrency    : {args.concurrency}")
    if args.endpoint == "batch":
        print(f"batch_size     : {args.batch_size}")
    if health:
        print(f"api_workers    : {health.get('num_workers')}")
        print(f"queue_depth    : {health.get('queue_depth')}")
    print("-" * 80)
    print(f"wall_time_s    : {elapsed_s:.3f}")
    print(f"items_per_s    : {completed_items / elapsed_s if elapsed_s else 0.0:.3f}")
    print(f"http_reqs_per_s: {len(ok_results) / elapsed_s if elapsed_s else 0.0:.3f}")
    print(f"errors         : {len(failed_results)}")

    if latencies:
        print("-" * 80)
        print("HTTP request latency, milliseconds")
        print(f"min            : {min(latencies):.2f}")
        print(f"mean           : {statistics.mean(latencies):.2f}")
        print(f"p50            : {percentile(latencies, 0.50):.2f}")
        print(f"p90            : {percentile(latencies, 0.90):.2f}")
        print(f"p95            : {percentile(latencies, 0.95):.2f}")
        print(f"p99            : {percentile(latencies, 0.99):.2f}")
        print(f"max            : {max(latencies):.2f}")

    if metrics_before and metrics_after:
        before_requests = metrics_before.get("total_requests", 0)
        after_requests = metrics_after.get("total_requests", 0)
        before_batches = metrics_before.get("total_batches", 0)
        after_batches = metrics_after.get("total_batches", 0)
        before_errors = metrics_before.get("total_errors", 0)
        after_errors = metrics_after.get("total_errors", 0)
        print("-" * 80)
        print("Server metric deltas")
        print(f"total_requests : {after_requests - before_requests}")
        print(f"total_batches  : {after_batches - before_batches}")
        print(f"total_errors   : {after_errors - before_errors}")

    if failed_results:
        print("-" * 80)
        print("First errors")
        for result in failed_results[:5]:
            print(f"- {result.error}")


async def main_async(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(args.concurrency, 1) + 4)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
    ) as client:
        health = await fetch_json(client, "/health")
        if health is None:
            raise SystemExit(f"Could not reach PRM API at {args.base_url}")

        if args.warmup > 0:
            warmup_items = make_items(args.warmup)
            if args.endpoint == "score":
                warmup_units: Sequence[dict[str, str] | List[dict[str, str]]] = warmup_items
            else:
                warmup_units = chunked(warmup_items, args.batch_size)
            await run_load(client, args.endpoint, warmup_units, args.concurrency)

        metrics_before = await fetch_json(client, "/metrics")

        items = make_items(args.requests)
        if args.endpoint == "score":
            work_units = items
        else:
            work_units = chunked(items, args.batch_size)

        started = time.perf_counter()
        results = await run_load(client, args.endpoint, work_units, args.concurrency)
        elapsed_s = time.perf_counter() - started

        metrics_after = await fetch_json(client, "/metrics")
        print_report(args, results, elapsed_s, health, metrics_before, metrics_after)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PRM API throughput.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--endpoint",
        choices=("score", "batch"),
        default="score",
        help="Use /score for many concurrent single-item requests or /score/batch.",
    )
    parser.add_argument("--requests", type=int, default=64, help="Total scored items.")
    parser.add_argument("--concurrency", type=int, default=8, help="In-flight HTTP requests.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Items per /score/batch request.",
    )
    parser.add_argument("--warmup", type=int, default=4, help="Warmup scored items.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

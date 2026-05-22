#!/usr/bin/env python3
import argparse
import concurrent.futures
import statistics
import time
from openai import OpenAI


def run_one(client, model, prompt, max_tokens, temperature):
    start = time.perf_counter()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )

    end = time.perf_counter()

    text = resp.choices[0].message.content or ""

    usage = getattr(resp, "usage", None)
    completion_tokens = usage.completion_tokens if usage else None
    prompt_tokens = usage.prompt_tokens if usage else None
    total_tokens = usage.total_tokens if usage else None

    elapsed = end - start

    if completion_tokens is None:
        completion_tokens = len(text.split())

    return {
        "elapsed": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
        "chars": len(text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Example: http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt",
        default="Write a detailed explanation of how neural networks learn. Continue until the token limit.",
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print(f"Base URL:     {args.base_url}")
    print(f"Model:        {args.model}")
    print(f"Requests:     {args.requests}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Max tokens:   {args.max_tokens}")
    print()

    results = []

    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                client,
                args.model,
                args.prompt,
                args.max_tokens,
                args.temperature,
            )
            for _ in range(args.requests)
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{i}/{args.requests}] "
                    f"{result['completion_tokens']} tokens, "
                    f"{result['elapsed']:.2f}s, "
                    f"{result['tokens_per_second']:.2f} tok/s"
                )
            except Exception as e:
                print(f"[{i}/{args.requests}] ERROR: {e}")

    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    if not results:
        print("No successful requests.")
        return

    per_request_tps = [r["tokens_per_second"] for r in results]
    completion_tokens = [r["completion_tokens"] for r in results]
    latencies = [r["elapsed"] for r in results]

    total_completion_tokens = sum(completion_tokens)

    print()
    print("=" * 60)
    print("Benchmark results")
    print("=" * 60)
    print(f"Successful requests:        {len(results)} / {args.requests}")
    print(f"Wall time:                  {wall_time:.2f}s")
    print(f"Total completion tokens:    {total_completion_tokens}")
    print(f"Overall throughput:         {total_completion_tokens / wall_time:.2f} tok/s")
    print()
    print(f"Per-request tok/s avg:      {statistics.mean(per_request_tps):.2f}")
    print(f"Per-request tok/s median:   {statistics.median(per_request_tps):.2f}")
    print(f"Per-request tok/s min:      {min(per_request_tps):.2f}")
    print(f"Per-request tok/s max:      {max(per_request_tps):.2f}")
    print()
    print(f"Latency avg:                {statistics.mean(latencies):.2f}s")
    print(f"Latency median:             {statistics.median(latencies):.2f}s")
    print(f"Latency min:                {min(latencies):.2f}s")
    print(f"Latency max:                {max(latencies):.2f}s")


if __name__ == "__main__":
    main()

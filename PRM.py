from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_PRM_API_URL = os.environ.get(
    "PRM_API_URL",
    os.environ.get("PRM_BASE_URL", "https://prm.24102006.xyz"),
)

STEP_TAG = "ки"


@dataclass
class PRMInput:
    question: str
    output: str
    estimated_tokens: int
    num_steps: int


class PRM:
    """
    Thin compatibility wrapper around the PRM HTTP API.

    The old training loop expects two methods:
    - covert_to_input(problem, thoughts)
    - get_step_scores(input_for_prm) -> (step_scores, n_tokens)

    This class preserves that surface while moving scoring out of the training
    process and into the FastAPI service in prm_gguf/prm_api.py.
    """

    def __init__(
        self,
        PRM_name: str | None = None,
        device: str | None = None,
        max_length: int = 4096,
        api_base_url: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        healthcheck: bool = True,
    ) -> None:
        if api_base_url is None and PRM_name and PRM_name.startswith(("http://", "https://")):
            api_base_url = PRM_name

        self.PRM_name = PRM_name
        self.device = device
        self.max_length = max_length
        self.max_chars = max_length * 4
        self.api_base_url = (api_base_url or DEFAULT_PRM_API_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = httpx.Client(
            base_url=self.api_base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=16),
        )

        if healthcheck:
            self.health()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PRM":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        response = self.client.get("/health")
        response.raise_for_status()
        return response.json()

    def covert_to_input(self, problem: str, thoughts: list[str]) -> PRMInput:
        return self.convert_to_input(problem, thoughts)

    def convert_to_input(self, problem: str, thoughts: list[str]) -> PRMInput:
        if not thoughts:
            return PRMInput(
                question=problem,
                output="",
                estimated_tokens=self._estimate_tokens(problem),
                num_steps=0,
            )

        discard = 0
        while True:
            output = self._format_steps(thoughts[discard:], step_offset=discard)
            estimated_tokens = self._estimate_tokens(f"{problem} {output}")
            if estimated_tokens <= self.max_length or discard >= len(thoughts) - 1:
                break
            discard += 1

        return PRMInput(
            question=problem,
            output=output,
            estimated_tokens=estimated_tokens,
            num_steps=len(thoughts) - discard,
        )

    def get_step_scores(self, input_for_prm: PRMInput | dict[str, Any] | tuple[str, str]):
        prm_input = self._normalise_input(input_for_prm)
        if prm_input.num_steps == 0:
            return [], prm_input.estimated_tokens

        data = self._request_score(
            {
                "question": prm_input.question,
                "output": prm_input.output,
            }
        )
        return [float(score) for score in data["step_scores"]], prm_input.estimated_tokens

    def score_steps(self, problem: str, thoughts: list[str]) -> tuple[list[float], int]:
        return self.get_step_scores(self.convert_to_input(problem, thoughts))

    def _request_score(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post("/score", json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff * attempt)

        raise RuntimeError(
            f"PRM API request failed after {self.max_retries} attempts at "
            f"{self.api_base_url}"
        ) from last_error

    @staticmethod
    def _format_steps(thoughts: list[str], step_offset: int = 0) -> str:
        return "\n".join(
            f"Step {idx + 1 + step_offset}: {thought} {STEP_TAG}"
            for idx, thought in enumerate(thoughts)
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 4)

    def _normalise_input(self, input_for_prm: PRMInput | dict[str, Any] | tuple[str, str]) -> PRMInput:
        if isinstance(input_for_prm, PRMInput):
            return input_for_prm

        if isinstance(input_for_prm, dict):
            question = input_for_prm["question"]
            output = input_for_prm["output"]
            return PRMInput(
                question=question,
                output=output,
                estimated_tokens=self._estimate_tokens(f"{question} {output}"),
                num_steps=output.count(STEP_TAG),
            )

        if isinstance(input_for_prm, tuple) and len(input_for_prm) == 2:
            question, output = input_for_prm
            return PRMInput(
                question=question,
                output=output,
                estimated_tokens=self._estimate_tokens(f"{question} {output}"),
                num_steps=output.count(STEP_TAG),
            )

        raise TypeError(
            "input_for_prm must be PRMInput, {'question', 'output'}, or "
            "(question, output)"
        )


if __name__ == "__main__":
    question = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast and "
        "bakes muffins with four. She sells the rest for $2 each. How much?"
    )
    thoughts = [
        "Janet starts with 16 eggs.",
        "After eating 3 and baking with 4, she has 16 - 3 - 4 = 9 eggs.",
        "She sells 9 eggs at $2 each, so she makes 9 * 2 = 18 dollars.",
    ]

    reward_model = PRM()
    scores, n_tokens = reward_model.score_steps(question, thoughts)
    print("estimated_tokens:", n_tokens)
    print("step_scores:", scores)

"""Echo task — the smallest self-improvement demo.

The operator must reply with a single JSON object: {"echo": "<the keyword>"}.
Graded deterministically. A fresh model with a bare harness will often wrap the
answer in prose or drop the key; the refiner learns to put a JSON-format
instruction in the system prompt and the score jumps. Fast, cheap, no domain
knowledge — the canonical smoke test for the loop.

Run the loop on it with:  python run_loop.py config/echo.json
"""
from __future__ import annotations

import json
import random

from . import Grade, register


@register("echo")
def make_echo(spec: dict):
    keywords = spec.get("keywords") or ["apple", "bridge", "candle", "delta", "ember", "flint"]
    return EchoTask(list(keywords))


class EchoTask:
    name = "echo"
    grading = "deterministic"

    def __init__(self, keywords: list[str]):
        self.keywords = keywords

    def generate_queries(self, n: int) -> list[str]:
        return [random.choice(self.keywords) for _ in range(n)]

    def reference(self, query: str) -> str:
        return query

    def evaluate(self, query: str, output: str, reference: str) -> Grade:
        try:
            start = output.find("{")
            end = output.rfind("}")
            obj = json.loads(output[start:end + 1]) if start != -1 and end > start else {}
        except json.JSONDecodeError:
            obj = {}
        ok = isinstance(obj, dict) and obj.get("echo") == reference
        return Grade(ok, 1.0 if ok else 0.0,
                     "matched" if ok else f"expected {{\"echo\": {reference!r}}}")


if __name__ == "__main__":
    t = make_echo({})
    q = t.generate_queries(1)[0]
    print(q, "->", t.evaluate(q, json.dumps({"echo": q}), q))

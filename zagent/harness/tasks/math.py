"""Math task — small integer arithmetic, graded by an exact numeric check.

The operator must return just the number (e.g. "42"). A bare harness may give a
sentence ("The answer is 42") or explain the working; the evaluator tolerates a
trailing numeric answer anywhere in the output. This gives the refiner something
real to learn: a system prompt like "compute step by step, output the final
integer last" measurably raises accuracy on larger operands.

Run the loop on it with:  python run_loop.py config/math.json
"""
from __future__ import annotations

import random
import re

from . import Grade, register


@register("math")
def make_math(spec: dict):
    return MathTask(
        n=spec.get("n", 2),
        lo=spec.get("lo", 2),
        hi=spec.get("hi", 50),
        allow_negative=spec.get("allow_negative", True),
    )


class MathTask:
    name = "math"
    grading = "deterministic"

    def __init__(self, n: int, lo: int, hi: int, allow_negative: bool):
        self.n = n
        self.lo = lo
        self.hi = hi
        self.allow_negative = allow_negative
        self._ops = ["+", "-", "*"] if allow_negative else ["+", "*"]

    def generate_queries(self, n: int) -> list[str]:
        out = []
        for _ in range(n):
            terms = [random.randint(self.lo, self.hi) for _ in range(self.n)]
            op = random.choice(self._ops)
            expr = f" {op} ".join(str(t) for t in terms)
            out.append(f"Compute: {expr}")
        return out

    def reference(self, query: str) -> str:
        expr = query.split("Compute:", 1)[1].strip()
        # Safe: terms are small integers and ops are + - * only.
        return str(eval(expr))  # noqa: S307  (controlled expression)

    def evaluate(self, query: str, output: str, reference: str) -> Grade:
        nums = [int(x) for x in re.findall(r"-?\d+", output)]
        if not nums:
            return Grade(False, 0.0, "no number found in output")
        got = nums[-1]  # last number is the usual "final answer" position
        ok = got == int(reference)
        return Grade(ok, 1.0 if ok else 0.0, f"expected {reference}, got {got}")


if __name__ == "__main__":
    t = make_math({})
    q = t.generate_queries(1)[0]
    print(q, "=", t.reference(q))

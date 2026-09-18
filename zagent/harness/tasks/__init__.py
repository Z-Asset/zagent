"""Task protocol + registry.

A "task" is the thing the operator tries to solve and the evaluator grades. The
loop is task-agnostic: a task is an object with four methods. Drop a new task in
`alpharsi/tasks/` and register it in TASK_REGISTRY to run the loop on it.

The evaluator here is DETERMINISTIC (closed). For open-ended tasks with no
checker, set the task's `grading` to "llm" and the kernel will route grading to
the evaluator model instead — but the evaluator model's source is still never
shown to the refiner, so the boundary holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class Grade:
    succeeded: bool
    metric: float            # 0..1, higher is better
    diagnostics: str = ""


class Task(Protocol):
    name: str
    grading: str             # "deterministic" or "llm"

    def generate_queries(self, n: int) -> list[str]: ...
    def reference(self, query: str) -> str: ...
    def evaluate(self, query: str, output: str, reference: str) -> Grade: ...


TASK_REGISTRY: dict[str, Callable[[dict], Task]] = {}


def register(name: str):
    def decorator(factory: Callable[[dict], Task]):
        TASK_REGISTRY[name] = factory
        return factory
    return decorator


def make_task(spec: dict) -> Task:
    name = spec.get("name")
    if not name or name not in TASK_REGISTRY:
        raise ValueError(f"unknown task {name!r} (available: {sorted(TASK_REGISTRY)})")
    return TASK_REGISTRY[name](spec)


# Import built-in tasks so their @register decorators run. Add a new task here.
from . import echo  # noqa: E402,F401
from . import math  # noqa: E402,F401

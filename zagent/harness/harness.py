"""Deterministic harness mutator.

The refiner only PROPOSES patches; this module decides whether and how to apply
them, and it enforces the closed-evaluator boundary (there is no operation that
touches the evaluator or the test set). It is deliberately dumb and total: every
operation either applies cleanly or raises, no partial state.
"""
from __future__ import annotations

from .types import HarnessState, PatchRecord, PatchSpec


def apply_patch(harness: HarnessState, patch: PatchRecord) -> HarnessState:
    """Return a NEW harness with the patch applied (the original is untouched)."""
    patch.validate()
    state = harness.snapshot()
    for op in patch.ops:
        _apply_op(state, op)
    return state


def _apply_op(state: HarnessState, op: PatchSpec) -> None:
    path = op.path
    if path == "system_prompt":
        if op.operation != "replace":
            raise ValueError("system_prompt only supports replace")
        state.system_prompt = op.value
    elif path == "memory":
        if op.operation == "add":
            state.memory.append(op.value)
        elif op.operation == "replace":
            state.memory = [line for line in op.value.splitlines() if line.strip()]
        elif op.operation == "remove":
            if op.index is None:
                raise ValueError("memory remove requires index")
            state.memory.pop(op.index)
        elif op.operation == "edit":
            if op.index is None:
                raise ValueError("memory edit requires index")
            state.memory[op.index] = op.value
        else:
            raise ValueError(f"bad memory op {op.operation}")
    elif path.startswith("skills."):
        name = path[len("skills."):]
        if not name:
            raise ValueError("skills path requires a name")
        if op.operation == "add":
            if name in state.skills:
                raise ValueError(f"skill {name!r} already exists; use replace")
            state.skills[name] = op.value
        elif op.operation == "replace":
            state.skills[name] = op.value
        elif op.operation == "remove":
            state.skills.pop(name, None)
        elif op.operation == "edit":
            raise ValueError("skill edit not supported; use replace")
        else:
            raise ValueError(f"bad skill op {op.operation}")
    else:
        raise ValueError(f"unknown harness path {path}")

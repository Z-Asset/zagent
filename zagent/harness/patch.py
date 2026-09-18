"""Parse and validate a refiner's proposed patch.

The refiner returns JSON (possibly wrapped in prose); this module turns it into
a typed PatchRecord or raises a clear error. The error is surfaced to the kernel,
which just logs it and keeps the previous harness — a malformed proposal is never
half-applied.
"""
from __future__ import annotations

import json

from .types import PatchRecord, PatchSpec


def parse_patch(text: str, patch_id: str) -> PatchRecord:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"refiner output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("refiner JSON must be an object")

    ops_raw = data.get("operations")
    if not isinstance(ops_raw, list):
        raise ValueError("patch requires an 'operations' list")

    ops: list[PatchSpec] = []
    for i, item in enumerate(ops_raw):
        if not isinstance(item, dict):
            raise ValueError(f"operation #{i} is not an object")
        path = item.get("path")
        if not isinstance(path, str):
            raise ValueError(f"operation #{i} missing string 'path'")
        operation = item.get("operation")
        if not isinstance(operation, str):
            raise ValueError(f"operation #{i} missing string 'operation'")
        value = item.get("value", "")
        if not isinstance(value, str):
            raise ValueError(f"operation #{i} 'value' must be a string")
        index = item.get("index")
        if index is not None and not isinstance(index, int):
            raise ValueError(f"operation #{i} 'index' must be an int or omitted")
        ops.append(PatchSpec(path=path, operation=operation, value=value, index=index))

    hypothesis = data.get("hypothesis", "")
    if not isinstance(hypothesis, str):
        raise ValueError("'hypothesis' must be a string")

    fail_causes = data.get("fail_causes", [])
    if not isinstance(fail_causes, list) or not all(isinstance(f, str) for f in fail_causes):
        raise ValueError("'fail_causes' must be a list of strings")

    expected = data.get("expected_effect", "")
    risk = data.get("risk", "")
    targeted = data.get("targeted_trace_ids", [])

    return PatchRecord(
        patch_id=patch_id,
        hypothesis=hypothesis,
        fail_causes=tuple(fail_causes),
        ops=ops,
        expected_effect=expected if isinstance(expected, str) else "",
        risk=risk if isinstance(risk, str) else "",
        targeted_trace_ids=[t for t in targeted if isinstance(t, str)] if isinstance(targeted, list) else [],
    )


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of prose the model may have wrapped around it."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in refiner output")
    return text[start:end + 1]

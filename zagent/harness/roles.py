"""The three roles of the loop.

  Operator  — runs the task against the current harness (the "Agent").
  Refiner   — reads the evaluator's learning signals and PROPOSES harness patches.
  Evaluator — grades operator output; deterministic by default, and CLOSED.

Roles are thin wrappers around a ChatModel plus their prompt templates. Keeping
them here (rather than buried in the kernel) makes it obvious what each role can
and cannot touch: the operator only sees the harness, the refiner only sees
signals + the current harness, the evaluator only sees (query, output, reference).
"""
from __future__ import annotations

from .harness import apply_patch
from ..models import ChatModel, build_harness_messages
from .patch import parse_patch
from .tasks import Grade, Task
from .types import HarnessState, LearningSignal, PatchRecord

# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

OPERATOR_SYSTEM = (
    "You are an operator solving one task at a time. Follow the harness "
    "instructions exactly. Return your answer and nothing else."
)


class Operator:
    def __init__(self, model: ChatModel):
        self.model = model

    def run(self, harness: HarnessState, query: str) -> str:
        messages = build_harness_messages(harness, query)
        return self.model.complete(messages).strip()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Closed: holds a Task, never exposes its internals to the refiner.

    When the task grades deterministically the model is unused. When a task is
    `grading = "llm"` the evaluator model is consulted, but only (query, output,
    reference) reach it — never the harness, never the refiner's proposals.
    """

    def __init__(self, task: Task, model: ChatModel | None = None):
        self.task = task
        self.model = model

    def grade(self, query: str, output: str) -> tuple[Grade, str]:
        reference = self.task.reference(query)
        if self.task.grading == "llm" and self.model is not None:
            return self._llm_grade(query, output, reference), reference
        return self.task.evaluate(query, output, reference), reference

    def _llm_grade(self, query: str, output: str, reference: str) -> Grade:
        prompt = (
            "Grade this answer against the reference. Respond with JSON only:\n"
            '{"succeeded": bool, "metric": 0..1, "diagnostics": "..."}\n\n'
            f"Query: {query}\nOutput: {output}\nReference: {reference}"
        )
        import json

        text = self.model.complete([{"role": "user", "content": prompt}], json_mode=True)
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            return Grade(False, 0.0, f"unparseable llm grade: {text[:200]}")
        return Grade(
            bool(obj.get("succeeded")),
            float(obj.get("metric", 1.0 if obj.get("succeeded") else 0.0)),
            str(obj.get("diagnostics", "")),
        )


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

REFINER_SYSTEM = (
    "You are a harness refiner. Given recent execution traces and the current "
    "harness, diagnose what went wrong and propose ONE patch as a JSON object:\n"
    '{"hypothesis": "...", "fail_causes": ["reasoning", "knowledge", "verification", "distraction"], '
    '"operations": [{"path": "system_prompt"|"memory"|"skills.<name>", "operation": "add"|"replace"|"remove"|"edit", '
    '"value": "...", "index": optional_int}], "expected_effect": "...", "risk": "..."}\n\n'
    "Rules: do not touch the evaluator or test set. Prefer the smallest edit that "
    "fixes the failure. If nothing is clearly wrong, return an empty operations list."
)


class Refiner:
    def __init__(self, model: ChatModel):
        self.model = model

    def propose(self, signals: list[LearningSignal], harness: HarnessState, patch_id: str) -> PatchRecord:
        failures = [
            f"- [{s.trace_id}] {'PASS' if s.succeeded else 'FAIL'} query={s.query!r}\n"
            f"  output={s.output!r}\n  diagnostics={s.diagnostics!r}"
            for s in signals
        ]
        user = (
            "Recent traces:\n" + "\n".join(failures)
            + "\n\nCurrent harness system_prompt:\n" + harness.system_prompt
            + f"\n\nMemory ({len(harness.memory)} lines), Skills ({list(harness.skills)}).\n"
            "Propose the patch JSON now."
        )
        text = self.model.complete(
            [{"role": "system", "content": REFINER_SYSTEM}, {"role": "user", "content": user}],
            json_mode=True,
        )
        return parse_patch(text, patch_id)


__all__ = ["Operator", "Evaluator", "Refiner", "apply_patch"]

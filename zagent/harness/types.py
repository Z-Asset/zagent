"""Core datatypes for the RSI-Harness self-improvement loop.

The loop is an online (never-reset) cycle of three roles — Operator, Refiner,
Evaluator — that together turn a model's own execution traces into a
self-improving harness (system prompt + memory + skills). This module holds the
plain data structures the three roles pass around; it has no logic.

Design notes (MetaRSI-v1 / Continual Harness, arXiv:2605.09998):
  - The harness (H) is the only thing the loop edits. Model weights are frozen.
  - The evaluator is a CLOSED artefact: the refiner never sees its source and
    may not edit it. That is what keeps a "performance gain" from being a trick
    that rewrites the grading standard.
  - Every mutation carries a hypothesis and a risk note, so the loop is
    auditable and a bad patch can be rejected or reverted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailCause(str, Enum):
    """Four failure causes a patch may claim to target (Data-RSI signature)."""
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    DISTRACTION = "distraction"


@dataclass(frozen=True)
class LearningSignal:
    """One attempt, compiled by the evaluator from an operator output.

    This is the object the refiner reads to decide what (if anything) to change.
    """
    trace_id: str
    query: str
    output: str
    succeeded: bool
    metric: float | None = None          # raw evaluator metric (0..1), if the task yields one
    diagnostics: str = ""                # plain-language notes on why it failed / what wobbled
    reference: str = ""                  # ground truth, when the evaluator exposes it
    fail_causes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.succeeded


@dataclass
class HarnessState:
    """The current system prompt + memory + skills.

    This is the H in S = (D, theta, H). Immutable-by-convention: callers produce
    a new instance via harness.apply_patch rather than mutating fields directly.
    """
    system_prompt: str
    memory: list[str] = field(default_factory=list)
    skills: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> "HarnessState":
        return HarnessState(
            system_prompt=self.system_prompt,
            memory=list(self.memory),
            skills=dict(self.skills),
        )


@dataclass
class PatchSpec:
    """A single edit to the harness (one patch = one slot change).

    `path` is a dotted locator: "system_prompt", "memory", or "skills.<name>".
    `operation` is one of add / replace / remove / edit.
    """
    path: str
    operation: str
    value: str = ""              # new content (for add / replace / edit)
    index: int | None = None     # memory index, when editing a specific line

    def validate(self) -> None:
        if self.path == "system_prompt":
            valid = {"replace"}
        elif self.path == "memory":
            valid = {"add", "replace", "remove", "edit"}
        elif self.path.startswith("skills.") and len(self.path) > len("skills."):
            valid = {"add", "replace", "remove", "edit"}
        else:
            raise ValueError(f"unknown harness path: {self.path}")
        if self.operation not in valid:
            raise ValueError(
                f"operation {self.operation!r} not valid for {self.path} (valid: {sorted(valid)})"
            )


@dataclass
class PatchRecord:
    """A harness patch as the refiner proposes it (before the kernel accepts it)."""
    patch_id: str
    hypothesis: str
    ops: list[PatchSpec]
    fail_causes: tuple[str, ...] = ()
    expected_effect: str = ""
    risk: str = ""
    targeted_trace_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("patch requires a hypothesis")
        if not self.ops:
            raise ValueError("patch requires at least one operation")
        for op in self.ops:
            op.validate()
        if self.expected_effect.strip() and not self.risk.strip():
            raise ValueError("patch with expected_effect must also declare risk")

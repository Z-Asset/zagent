"""The RSI kernel — the deterministic Observe→…→Select→Export loop.

This is the "no reset" cycle. The refiner only proposes; this module decides
whether to accept a patch (by replaying it against the recent failure slices and
keeping it only if the new harness strictly beats the current one). The evaluator
is closed: the kernel owns the Task and never hands it to the refiner.

Loop shape (online, never resets the environment):
  operator runs on the harness  →  evaluator grades  →  learning signals
  →  refiner proposes a patch  →  kernel validates on the failure slice
  →  keep only if strictly better  →  loop.

Usage (single-line):
  from zagent.harness.kernel import run_loop
  run_loop("config/echo.json")
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .harness import apply_patch
from ..models import ChatModel, LambdaModel, MockModel
from .patch import parse_patch
from .roles import Evaluator, Operator, Refiner
from .tasks import make_task
from .types import HarnessState, LearningSignal

# ---------------------------------------------------------------------------
# Config / model construction
# ---------------------------------------------------------------------------

def _model_from(spec: dict, default: ChatModel) -> ChatModel:
    """Build a ChatModel from a config entry, or fall back to the shared default."""
    if not spec:
        return default
    kind = spec.get("type", "openai")
    if kind == "mock":
        return MockModel(spec.get("response", "mock"))
    if kind == "lambda":
        raise ValueError("lambda models must be wired in code, not config")
    base_url = spec.get("base_url") or default.base_url
    api_key = spec.get("api_key") or default.api_key
    name = spec.get("model") or default.name
    temperature = spec.get("temperature", 0.0)
    return ChatModel(name, base_url=base_url, api_key=api_key, temperature=temperature)


def load_config(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def _offline_operator(messages):
    """Offline operator: follows the harness only if it already teaches JSON."""
    system = messages[0]["content"]
    user = messages[1]["content"]
    if "JSON" in system:
        return '{"echo": "' + user + '"}'
    return "The echo is " + user


def _offline_refiner(messages):
    user = messages[1]["content"]
    if "FAIL" not in user:
        # Nothing failed in the window -> propose no change, let the loop converge.
        return json.dumps({"hypothesis": "no failure observed", "fail_causes": [], "operations": []})
    return json.dumps({
        "hypothesis": "operator returns prose; it needs an explicit JSON-output instruction in the system prompt",
        "fail_causes": ["verification"],
        "operations": [{
            "path": "system_prompt",
            "operation": "replace",
            "value": 'You are an echo bot. Always output a JSON object {"echo": "<the keyword>"} and nothing else.',
        }],
        "expected_effect": "operator returns valid JSON and the echo task passes",
        "risk": "none; single-slot system prompt edit",
    })


def build_components(cfg: dict, offline: bool = False):
    """Construct (operator, evaluator, refiner, initial_harness, settings) from config."""
    if offline:
        task = make_task(cfg["task"])
        harness = HarnessState(
            system_prompt=cfg.get("harness", {}).get("system_prompt", "You are a helpful assistant."),
            memory=cfg.get("harness", {}).get("memory", []),
            skills=cfg.get("harness", {}).get("skills", {}),
        )
        return (
            Operator(LambdaModel(_offline_operator)),
            Evaluator(task),
            Refiner(LambdaModel(_offline_refiner)),
            harness,
            cfg.get("settings", {}),
        )

    model_cfg = cfg.get("model", {})
    default = ChatModel(
        model_cfg.get("model", "deepseek-chat"),
        base_url=model_cfg.get("base_url", "https://api.deepseek.com/v1"),
        api_key=model_cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", ""),
        temperature=model_cfg.get("temperature", 0.0),
    )
    operator_model = _model_from(cfg.get("operator", {}).get("model", {}), default)
    refiner_model = _model_from(cfg.get("refiner", {}).get("model", {}), default)
    evaluator_model = _model_from(cfg.get("evaluator", {}).get("model", {}), default)

    task = make_task(cfg["task"])
    harness = HarnessState(
        system_prompt=cfg.get("harness", {}).get("system_prompt", "You are a helpful assistant."),
        memory=cfg.get("harness", {}).get("memory", []),
        skills=cfg.get("harness", {}).get("skills", {}),
    )
    settings = cfg.get("settings", {})
    return (
        Operator(operator_model),
        Evaluator(task, evaluator_model if task.grading == "llm" else None),
        Refiner(refiner_model),
        harness,
        settings,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    queries_seen: int = 0
    successes: int = 0
    patches_proposed: int = 0
    patches_accepted: int = 0
    history: list[dict] = field(default_factory=list)

    def rolling_accuracy(self) -> float:
        return self.successes / self.queries_seen if self.queries_seen else 0.0


def _score(operator: Operator, evaluator: Evaluator, harness: HarnessState, query: str) -> tuple[float, LearningSignal]:
    output = operator.run(harness, query)
    grade, reference = evaluator.grade(query, output)
    signal = LearningSignal(
        trace_id=f"tr-{uuid.uuid4().hex[:8]}",
        query=query,
        output=output,
        succeeded=grade.succeeded,
        metric=grade.metric,
        diagnostics=grade.diagnostics,
        reference=reference,
    )
    return (grade.metric, signal)


def run_loop(config_path: str, offline: bool = False) -> RunStats:
    cfg = load_config(config_path)
    operator, evaluator, refiner, harness, settings = build_components(cfg, offline)

    epochs = int(settings.get("epochs", 10))
    queries_per_epoch = int(settings.get("queries_per_epoch", 5))
    refine_every = int(settings.get("refine_every", 1))
    refine_window = int(settings.get("refine_window", queries_per_epoch))
    replay_on_accept = settings.get("replay_on_accept", True)
    out_dir = Path(settings.get("out_dir", "runs"))

    stats = RunStats()
    recent_signals: list[LearningSignal] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    export_path = out_dir / run_id

    for epoch in range(1, epochs + 1):
        queries = evaluator.task.generate_queries(queries_per_epoch)
        epoch_signals: list[LearningSignal] = []
        for query in queries:
            metric, signal = _score(operator, evaluator, harness, query)
            stats.queries_seen += 1
            stats.successes += 1 if signal.succeeded else 0
            epoch_signals.append(signal)
            recent_signals.append(signal)
            stats.history.append({
                "epoch": epoch, "query": query, "output": signal.output,
                "succeeded": signal.succeeded, "metric": metric,
                "diagnostics": signal.diagnostics,
            })
            if cfg.get("verbose", False):
                print(f"[e{epoch}] {'PASS' if signal.succeeded else 'FAIL'} {query[:40]!r} -> {signal.output[:60]!r}")

        print(f"epoch {epoch}/{epochs}: acc={stats.rolling_accuracy():.3f} "
              f"({stats.successes}/{stats.queries_seen})")

        if epoch % refine_every == 0:
            window = recent_signals[-refine_window:]
            failed = [s for s in window if not s.succeeded]
            patch_id = f"p-{uuid.uuid4().hex[:8]}"
            stats.patches_proposed += 1
            try:
                patch = refiner.propose(window, harness, patch_id)
            except Exception as exc:  # malformed proposal -> keep current harness
                print(f"  refiner proposal rejected (parse/validate): {exc}")
                continue
            if not patch.ops:
                print(f"  refiner found nothing to change ({patch_id})")
                continue
            candidate = apply_patch(harness, patch)
            # Validate: only accept if strictly better on the recent failure slice.
            if replay_on_accept and failed:
                before = sum(1 for s in failed if s.succeeded)
                after = 0
                for s in failed:
                    _, sig = _score(operator, evaluator, candidate, s.query)
                    after += 1 if sig.succeeded else 0
                if after <= before:
                    print(f"  patch {patch_id} rejected: no strict gain on failure slice ({before}->{after})")
                    continue
            harness = candidate
            stats.patches_accepted += 1
            print(f"  patch {patch_id} accepted: {patch.hypothesis[:80]}")

    # Export.
    export_path.mkdir(parents=True, exist_ok=True)
    final_acc = stats.rolling_accuracy()
    (export_path / "harness.json").write_text(
        json.dumps({
            "system_prompt": harness.system_prompt,
            "memory": harness.memory,
            "skills": harness.skills,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_path / "history.jsonl").write_text(
        "\n".join(json.dumps(h, ensure_ascii=False) for h in stats.history) + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id, "queries_seen": stats.queries_seen, "successes": stats.successes,
        "final_accuracy": final_acc, "patches_proposed": stats.patches_proposed,
        "patches_accepted": stats.patches_accepted, "config": config_path,
    }
    (export_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"exported to {export_path}")
    return stats


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:]]
    offline = "--offline" in args
    args = [a for a in args if a != "--offline"]
    run_loop(args[0] if args else "config/echo.json", offline=offline)

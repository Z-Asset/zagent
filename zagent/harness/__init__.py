"""Harness-RSI 算子 —— 进化 harness（系统提示词 / 记忆 / 技能），不动模型权重。

三角色：Operator（按 harness 解任务）、Evaluator（封闭判分）、Refiner（读信号
提补丁）。主循环 `run_loop`。参照 MetaRSI-v1 的 Harness-RSI 算子与 Continual
Harness 三格骨架。
"""
from .kernel import run_loop, load_config, build_components
from .types import HarnessState, LearningSignal, PatchRecord, PatchSpec
from .roles import Operator, Evaluator, Refiner

__all__ = [
    "run_loop", "load_config", "build_components",
    "HarnessState", "LearningSignal", "PatchRecord", "PatchSpec",
    "Operator", "Evaluator", "Refiner",
]

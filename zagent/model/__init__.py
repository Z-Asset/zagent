"""Model-RSI 算子 —— 进化 alpha 模型的训练配方，不动 prompt/记忆。

三角色：Operator（框架内置 walk-forward）、Evaluator（OOS_RankIC 验收）、
Refiner（离线 TPE 贝叶斯搜索 / 在线 DeepSeek LLM）。主循环 `run_alpha`。
"""
from .alpha_rsi import run_alpha, run_alpha_loop, AlphaRunStats
from .alpha_eval import EvalResult, RECIPE_SPACE, evaluate_recipe, evaluate_recipe_isolated, validate_recipe
from .offline_refiner import OfflineTPERefiner
from .walkforward import run_walk_forward, expanding_windows
from .metrics import summarize, rank_ic_series

__all__ = [
    "run_alpha", "run_alpha_loop", "AlphaRunStats",
    "EvalResult", "RECIPE_SPACE", "evaluate_recipe", "evaluate_recipe_isolated",
    "validate_recipe", "OfflineTPERefiner",
    "run_walk_forward", "expanding_windows", "summarize", "rank_ic_series",
]

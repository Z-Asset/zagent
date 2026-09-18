"""Model-RSI 闭环 —— 三角色把 alpha 模型的训练配方进化出来。

走 Model-RSI 路线（改训练配方，不改 prompt/memory）。三角色映射：

  Operator  = alpha_eval.evaluate_recipe()   确定性训练脚本（非 LLM）
  Evaluator = EvalResult.primary_metric()    主指标 OOS_RankIC（封闭）
  Refiner   = DeepSeek LLM                   读历史+最佳，提议下一组配方

闭环（每轮）：
  历史配方 → Refiner 提议新配方 → validate_recipe 校验（不合法拒绝）
  → 快速评估（fast_epochs 压小）→ 只保留 OOS_RankIC 严格更优 → 循环

成本约束（关键）：CPU 单块 walk-forward 15 epochs 约 357 秒，无法大量
迭代。因此快速评估默认 fast_epochs=1、blocks=1、max_train=4000（约 6 秒
/次），几十轮试错几分钟内完成；最后再全量验证最佳配方。

离线模式（--offline）用确定性配方生成器代替 LLM，验证闭环逻辑零依赖可转。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .alpha_eval import (
    DEFAULT_RECIPE, RECIPE_SPACE, REQUIRED_KEYS, evaluate_recipe,
    evaluate_recipe_isolated, validate_recipe,
)
from ..models import ChatModel
from .offline_refiner import OfflineTPERefiner

# 非 TTY 下 stdout 行缓冲，让后台/管道运行时日志即时可见。
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Refiner prompt
# ---------------------------------------------------------------------------

REFINER_SYSTEM = (
    "你是 alpha 量化模型的训练配方调优器。目标：提高样本外 RankIC。\n"
    "给你配方合法空间、历史评估结果、当前最佳配方。请提议一组新的超参配方，"
    "返回 JSON：{\"recipe\": {...}, \"reason\": \"...\"}。\n\n"
    "规则：\n"
    "- 配方必须包含全部键：d_model, nhead, layers, ffn, dropout, epochs, lr, batch\n"
    "- d_model 必须能被 nhead 整除\n"
    "- 每个值必须落在合法空间内\n"
    "- 基于历史结果做有依据的试探（比如 RankIC 差就加大 d_model 或降 lr），"
    "不要随机乱试\n"
    "- 只返回 JSON，不要解释"
)


def _recipe_space_json() -> str:
    return json.dumps(RECIPE_SPACE, ensure_ascii=False)


def _history_json(history: list[dict]) -> str:
    rows = []
    for h in history:
        rows.append({
            "recipe": h["recipe"],
            "rank_ic": h.get("rank_ic"),
            "icir": h.get("icir"),
            "sr": h.get("sr"),
            "accepted": h.get("accepted", False),
        })
    return json.dumps(rows, ensure_ascii=False, indent=1)


class Refiner:
    """在线用 LLM；离线用 TPE 贝叶斯优化（同样智能，不靠 API）。"""

    def __init__(self, model: ChatModel | None, space: dict, dim_order: list[str],
                 seed: int, fixed: dict | None = None):
        self.model = model
        self._tpe = OfflineTPERefiner(space, dim_order, seed=seed, fixed=fixed)

    def propose(self, history: list[dict], best: dict | None, target_keys: list[str]) -> dict:
        if self.model is None:
            return self._tpe.propose(history)
        user = (
            f"配方合法空间：{_recipe_space_json()}\n\n"
            f"历史评估结果：{_history_json(history)}\n\n"
            f"当前最佳：{json.dumps(best, ensure_ascii=False) if best else '无'}\n\n"
            "请提议下一组配方。"
        )
        text = self.model.complete(
            [{"role": "system", "content": REFINER_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True,
        )
        return _extract_recipe(text)


def _extract_recipe(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("refiner 输出无 JSON")
    obj = json.loads(text[start:end + 1])
    recipe = obj.get("recipe", obj)
    if not isinstance(recipe, dict):
        raise ValueError("refiner JSON 缺 recipe")
    return recipe


# ---------------------------------------------------------------------------
# 闭环主循环
# ---------------------------------------------------------------------------

@dataclass
class AlphaRunStats:
    iterations: int = 0
    accepted: int = 0
    best_rank_ic: float = float("-inf")
    best_recipe: dict | None = None
    history: list[dict] = field(default_factory=list)


def _to_model(model_cfg: dict) -> ChatModel | None:
    """从配置建 LLM；api_key 留空读 DEEPSEEK_API_KEY。离线时返回 None。"""
    if not model_cfg:
        return None
    name = model_cfg.get("model", "deepseek-chat")
    base_url = model_cfg.get("base_url", "https://api.deepseek.com/v1")
    api_key = model_cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    return ChatModel(name, base_url=base_url, api_key=api_key,
                     temperature=model_cfg.get("temperature", 0.0))


def run_alpha_loop(config_path: str, offline: bool = False) -> AlphaRunStats:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    # pkg 支持相对路径：相对 config 所在目录解析，迁移后可写相对或绝对路径。
    pkg = Path(cfg["pkg"])
    if not pkg.is_absolute():
        pkg = (Path(config_path).resolve().parent / pkg).resolve()
    return run_alpha(pkg, cfg.get("settings", {}), offline=offline, config_path=config_path)


def run_alpha(
    pkg: Path,
    settings: dict | None = None,
    *,
    offline: bool = True,
    config_path: str = "",
) -> AlphaRunStats:
    """核心闭环，不依赖 config 文件。pkg 为数据包路径（含 pkg_api.py）。

    settings 键（缺省均有合理默认）：iterations / blocks / max_train /
    fast_epochs / threads / L / seed / out_dir / isolated / timeout_s /
    fixed_epochs / patience / n_seeds。
    """
    settings = settings or {}
    pkg = Path(pkg).resolve()
    iterations = int(settings.get("iterations", 10))
    blocks = settings.get("blocks")         # None = 全量
    max_train = int(settings.get("max_train", 4000))
    fast_epochs = settings.get("fast_epochs")   # None = 用配方真实 epochs
    threads = int(settings.get("threads", 2))
    L = int(settings.get("L", 60))
    seed = int(settings.get("seed", 42))
    n_seeds = int(settings.get("n_seeds", 1))
    out_dir = Path(settings.get("out_dir", "runs_alpha"))
    isolated = bool(settings.get("isolated", True))
    timeout_s = settings.get("timeout_s")   # None = 不超时；单轮评估建议设上限
    fixed_epochs = settings.get("fixed_epochs")   # 非 None 时把 epochs 固定、移出搜索空间

    fixed = {}
    if fixed_epochs is not None:
        fixed["epochs"] = int(fixed_epochs)

    refiner = Refiner(None if offline else _to_model(settings.get("model", {})),
                      RECIPE_SPACE, REQUIRED_KEYS, seed=seed, fixed=fixed)
    target_keys = REQUIRED_KEYS
    patience = int(settings.get("patience", 0))   # 0 = 不早停

    # 初始：先评估基线配方。
    stats = AlphaRunStats()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"alpha-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"[alpha-rsi] pkg={pkg}  iterations={iterations}  "
          f"fast_epochs={fast_epochs}  blocks={blocks}  max_train={max_train}  "
          f"patience={patience}  isolated={isolated}")

    # 进度条：共 1(基线) + iterations 个配方要评估
    total_steps = 1 + iterations
    done = 0

    def _progress():
        nonlocal done
        done += 1
        bar = "#" * done + "-" * (total_steps - done)
        print(f"  [进度 {done}/{total_steps}] [{bar}]")

    def _evaluate(recipe):
        if isolated:
            return evaluate_recipe_isolated(
                pkg, recipe, blocks=blocks, max_train=max_train, L=L, seed=seed,
                fast_epochs=fast_epochs, threads=threads, n_seeds=n_seeds,
                timeout_s=timeout_s)
        return evaluate_recipe(pkg, recipe, blocks=blocks, max_train=max_train,
                               fast_epochs=fast_epochs, L=L, seed=seed,
                               threads=threads, n_seeds=n_seeds)

    # 基线（套用 fixed，保证与候选公平对比：epochs 一致）
    base = dict(DEFAULT_RECIPE)
    base.update(fixed)
    print(f"[alpha-rsi] 评估基线配方 {base['d_model']}d/{base['layers']}L ep={base['epochs']} ...")
    res = _evaluate(base)
    stats.best_rank_ic = res.primary_metric()
    stats.best_recipe = dict(base)
    stats.history.append(_record(base, res, accepted=True, reason="基线"))
    _progress()

    no_improve = 0
    for it in range(1, iterations + 1):
        stats.iterations = it
        try:
            proposal = refiner.propose(stats.history, stats.best_recipe, target_keys)
            recipe = validate_recipe(proposal)
            reason = refiner.model is None and _explain(recipe, stats.best_recipe) or ""
        except Exception as exc:
            print(f"  [it {it}] 提议被拒（{exc}）")
            stats.history.append({"iteration": it, "rejected": str(exc)})
            continue

        print(f"  [it {it}] 评估 {recipe['d_model']}d/{recipe['nhead']}h/{recipe['layers']}L "
              f"lr={recipe['lr']} ep={recipe['epochs']} ...")
        try:
            res = _evaluate(recipe)
        except Exception as exc:
            print(f"      评估崩溃（已隔离，跳过本轮）: {exc}")
            stats.history.append({"iteration": it, "recipe": recipe,
                                  "crashed": str(exc)})
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  [早停] 连续 {patience} 轮无改进/崩溃，停止探索")
                break
            continue

        metric = res.primary_metric()
        accepted = metric > stats.best_rank_ic
        if accepted:
            stats.accepted += 1
            stats.best_rank_ic = metric
            stats.best_recipe = dict(recipe)
            no_improve = 0
        else:
            no_improve += 1
        stats.history.append(_record(recipe, res, accepted=accepted, reason=reason))

        verdict = "接受(更优)" if accepted else "拒绝(未超)"
        print(f"      RankIC={metric:+.4f}  ICIR={res.oos_icir}  SR={res.oos_sr}  "
              f"耗时={res.elapsed_s}s  -> {verdict}" + (f"  [{reason}]" if reason else ""))
        _progress()

        if patience and no_improve >= patience:
            print(f"  [早停] 连续 {patience} 轮无改进，停止探索")
            break

    # 导出
    out_path = out_dir / run_id
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "history.json").write_text(
        json.dumps(stats.history, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id, "iterations": stats.iterations, "accepted": stats.accepted,
        "best_rank_ic": stats.best_rank_ic, "best_recipe": stats.best_recipe,
        "pkg": str(pkg),
    }
    (out_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[alpha-rsi] 完成：最佳 RankIC={stats.best_rank_ic:.4f}")
    print(f"  最佳配方：{json.dumps(stats.best_recipe, ensure_ascii=False)}")
    print(f"  导出：{out_path}")
    return stats


def _explain(recipe: dict, best: dict | None) -> str:
    """离线 TPE 选点的可读理由（审计用）。"""
    if best is None:
        return "初始覆盖采样"
    diffs = [f"{k} {best.get(k)}->{recipe.get(k)}"
             for k in recipe if recipe.get(k) != best.get(k)]
    return "调整 " + ", ".join(diffs) if diffs else "在最优附近保持、局部探索"


def _record(recipe: dict, res, accepted: bool, reason: str) -> dict:
    return {
        "recipe": recipe,
        "rank_ic": res.oos_rank_ic,
        "icir": res.oos_icir,
        "sr": res.oos_sr,
        "r2": res.oos_r2,
        "n_days": res.n_days,
        "n_blocks": res.n_blocks,
        "elapsed_s": res.elapsed_s,
        "degenerate": res.degenerate,
        "accepted": accepted,
        "reason": reason,
    }


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--offline"]
    offline = "--offline" in sys.argv
    run_alpha_loop(args[0] if args else "config/alpha.json", offline=offline)

"""Alpha 模型评估器 —— 框架内置 walk-forward 协议，数据包只提供读数据 + 建模型。

这一层是 Model-RSI 的 Operator + Evaluator（确定性代码，不是 LLM）：

  Operator  = walkforward.run_walk_forward()   框架内置 expanding walk-forward
  Evaluator = metrics.summarize()              算 OOS_RankIC / OOS_ICIR / OOS_SR

数据包（pkg）须暴露 pkg_api.py，实现两个接口：
  read_data(pkg) -> (X, R, dates, codes)   已预处理的 (T,N,K) + (T,N) + 交易日 + 代码
  build_model(recipe, K, L, seed) -> 有 fit/predict 的对象

框架自己保证：窗口切分（2 天空白）、序列切窗（无前视）、训练/验证/测试、
多 seed 降噪、指标口径。数据包不再自己实现 walk-forward。

配方必须过 validate_recipe 校验（约束到合法空间），不合法直接拒绝。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 非 TTY 下 stdout 行缓冲。
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# 配方合法空间（trans 模型）。Refiner 的提议必须落在这些约束内。
# ---------------------------------------------------------------------------

RECIPE_SPACE: dict[str, Any] = {
    "d_model": [16, 32, 48, 64, 96, 128],
    "nhead": [2, 4, 8],
    "layers": [1, 2, 3],
    "ffn": [64, 128, 256, 512],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "epochs": [3, 5, 10, 15],
    "lr": [0.0003, 0.001, 0.003],
    "batch": [1024, 2048, 4096],
}

# 默认基线：与 GRIDS['trans'] 现有候选一致。
DEFAULT_RECIPE = {"d_model": 32, "nhead": 4, "layers": 1, "ffn": 128,
                  "dropout": 0.1, "epochs": 15, "lr": 0.001, "batch": 4096}

REQUIRED_KEYS = ["d_model", "nhead", "layers", "ffn", "dropout", "epochs", "lr", "batch"]


def validate_recipe(recipe: dict) -> dict:
    """校验并补全一个配方。不合法抛 ValueError（调用方拒绝该提议）。"""
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    out: dict = {}
    for k in REQUIRED_KEYS:
        if k not in recipe:
            raise ValueError(f"recipe missing key {k!r}")
        v = recipe[k]
        if k in ("epochs", "batch"):
            v = int(v)
        if k in ("d_model", "ffn"):
            v = int(v)
        if k in ("nhead", "layers"):
            v = int(v)
        if k == "dropout":
            v = float(v)
        if k == "lr":
            v = float(v)
        out[k] = v
    # 结构约束
    if out["d_model"] % out["nhead"] != 0:
        raise ValueError(f"d_model({out['d_model']}) must be divisible by nhead({out['nhead']})")
    for k, v in out.items():
        if k in RECIPE_SPACE and isinstance(RECIPE_SPACE[k], list) and v not in RECIPE_SPACE[k]:
            raise ValueError(f"{k}={v} not in allowed space {RECIPE_SPACE[k]}")
    return out


def _load_api(pkg: Path):
    """从包路径动态加载 pkg_api.py，不污染 sys.path。

    pkg_api.py 是数据包对框架的标准接口，须暴露：
      read_data(pkg) -> (X, R, dates, codes)
      build_model(recipe, K, L, seed) -> fit/predict 对象
    """
    entry = pkg / "pkg_api.py"
    if not entry.exists():
        raise FileNotFoundError(
            f"数据包缺 pkg_api.py（需实现 read_data / build_model 两个接口）: {entry}")
    spec = importlib.util.spec_from_file_location("zagent_pkg_api", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {entry}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["zagent_pkg_api"] = mod
    spec.loader.exec_module(mod)
    for fn in ("read_data", "build_model"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError(f"pkg_api.py 缺必需接口 {fn}()")
    return mod


@dataclass
class EvalResult:
    """一次配方集合评估的归一化结果。"""
    recipe: dict
    oos_sr: float | None
    oos_rank_ic: float | None
    oos_icir: float | None
    oos_r2: float | None
    n_days: int
    n_blocks: int
    elapsed_s: float
    degenerate: bool = False
    raw: dict = field(default_factory=dict)

    def primary_metric(self) -> float:
        """主验收指标：OOS_RankIC（纯信号质量），NaN 记为 -inf。"""
        if self.oos_rank_ic is None:
            return float("-inf")
        return float(self.oos_rank_ic)

    def to_jsonable(self) -> dict:
        return {
            "recipe": self.recipe,
            "oos_sr": self.oos_sr,
            "oos_rank_ic": self.oos_rank_ic,
            "oos_icir": self.oos_icir,
            "oos_r2": self.oos_r2,
            "n_days": self.n_days,
            "n_blocks": self.n_blocks,
            "elapsed_s": self.elapsed_s,
            "degenerate": self.degenerate,
            "raw": self.raw,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "EvalResult":
        return cls(
            recipe=d["recipe"], oos_sr=d["oos_sr"], oos_rank_ic=d["oos_rank_ic"],
            oos_icir=d["oos_icir"], oos_r2=d["oos_r2"], n_days=d["n_days"],
            n_blocks=d["n_blocks"], elapsed_s=d["elapsed_s"],
            degenerate=d.get("degenerate", False), raw=d.get("raw", {}),
        )


# ---------------------------------------------------------------------------
# 评估入口
# ---------------------------------------------------------------------------

def evaluate_recipe(
    pkg: Path,
    recipe: dict,
    *,
    blocks: int = 1,
    max_train: int = 4000,
    L: int = 60,
    seed: int = 42,
    fast_epochs: int | None = None,
    threads: int = 2,
    n_seeds: int = 1,
    verbose: bool = False,
) -> EvalResult:
    """评估单个配方：框架内置 walk-forward（不依赖数据包自己实现）。

    数据包须暴露 pkg_api.py，含：
      read_data(pkg) -> (X, R, dates, codes)    已预处理的 (T,N,K) + (T,N) + 交易日 + 代码
      build_model(recipe, K, L, seed) -> 有 fit(X_seq, y)/predict(X_seq) 的对象

    fast_epochs 非 None 时覆盖配方里的 epochs（快速评估）；
    n_seeds > 1 时多 seed 跑取均值降噪（随机数多设置）。
    """
    import torch
    torch.set_num_threads(threads)

    api = _load_api(pkg)
    recipe = validate_recipe(recipe)
    if fast_epochs is not None:
        recipe = {**recipe, "epochs": int(fast_epochs)}

    X, R, dates, codes = api.read_data(pkg)
    from .walkforward import run_walk_forward

    t0 = time.time()
    result = run_walk_forward(
        X, R, dates, codes,
        build_model=api.build_model,
        L=L, recipe=recipe,
        seed=seed, max_train=max_train,
        max_blocks=blocks, verbose=verbose,
        n_seeds=n_seeds,
    )
    m = result["metrics"]

    def _f(v):
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    return EvalResult(
        recipe=recipe,
        oos_sr=_f(m.get("OOS_SR")),
        oos_rank_ic=_f(m.get("OOS_RankIC")),
        oos_icir=_f(m.get("OOS_ICIR")),
        oos_r2=_f(m.get("OOS_R2")),
        n_days=int(m.get("n_days", 0)),
        n_blocks=int(result["n_blocks"]),
        elapsed_s=round(time.time() - t0, 1),
        degenerate=False,
        raw={"OOS_RankIC_std": m.get("OOS_RankIC_std"), "n_seeds": n_seeds},
    )


def evaluate_recipe_isolated(pkg: Path, recipe: dict, *, blocks: int = 1,
                             max_train: int = 4000, L: int = 60, seed: int = 42,
                             fast_epochs: int | None = None, threads: int = 2,
                             n_seeds: int = 1,
                             timeout_s: float | None = None) -> EvalResult:
    """在独立子进程里评估，隔离 segfault / 内存累积。

    torch 2.13 CPU 版在 Windows + Python 3.14 下多线程不稳定，单轮评估可能
    segfault（exit 139）或超时。子进程隔离后，崩溃只杀掉 worker，主循环
    捕获并继续下一轮，不中断整个闭环。做法与 Optuna 处理不稳定 objective 一致。
    """
    import subprocess

    payload = json.dumps({
        "pkg": str(pkg), "recipe": recipe, "blocks": blocks, "max_train": max_train,
        "L": L, "seed": seed, "fast_epochs": fast_epochs, "threads": threads,
        "n_seeds": n_seeds,
    })
    cmd = [sys.executable, "-m", "zagent.model.alpha_eval", "--worker", "--pkg", str(pkg)]
    # cwd 必须是 zagent 包所在的父目录（项目根），
    # 否则 `-m zagent.model.alpha_eval` 找不到模块。
    # alpha_eval.py 在 <root>/zagent/model/ 下，上溯三级到 <root>。
    project_root = Path(__file__).resolve().parent.parent.parent
    proc = subprocess.run(
        cmd, input=payload, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s,
        cwd=str(project_root),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"worker 崩溃 rc={proc.returncode}（隔离已捕获，跳过本轮）: {' | '.join(tail)}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return EvalResult.from_jsonable(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
    raise RuntimeError("worker 无有效结果输出")


def _worker_main():
    """子进程入口：读 stdin 的 JSON 参数，跑评估，stdout 最后一行输出结果 JSON。"""
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--pkg", default=None)
    args, _ = ap.parse_known_args()
    payload = json.loads(sys.stdin.read())
    pkg = Path(payload["pkg"])
    res = evaluate_recipe(
        pkg, payload["recipe"], blocks=int(payload.get("blocks", 1)),
        max_train=int(payload.get("max_train", 4000)), L=int(payload.get("L", 60)),
        seed=int(payload.get("seed", 42)),
        fast_epochs=payload.get("fast_epochs"), threads=int(payload.get("threads", 2)),
        n_seeds=int(payload.get("n_seeds", 1)),
        verbose=False,
    )
    print(json.dumps(res.to_jsonable(), ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    if "--worker" in sys.argv:
        _worker_main()
        sys.exit(0)

    ap = argparse.ArgumentParser(description="Alpha 配方评估器（单独验证训练链路）")
    ap.add_argument("--pkg", default=".")
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--fast-epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--recipe", default=None, help="JSON 文件或内联 JSON，缺省用 DEFAULT_RECIPE")
    args = ap.parse_args()

    recipe = DEFAULT_RECIPE
    if args.recipe:
        txt = args.recipe
        if Path(args.recipe).exists():
            txt = Path(args.recipe).read_text(encoding="utf-8")
        recipe = json.loads(txt)

    res = evaluate_recipe(Path(args.pkg), recipe, blocks=args.blocks,
                          max_train=args.max_train, fast_epochs=args.fast_epochs,
                          threads=args.threads, verbose=True)
    print(json.dumps({
        "recipe": res.recipe,
        "OOS_RankIC": res.oos_rank_ic,
        "OOS_ICIR": res.oos_icir,
        "OOS_SR": res.oos_sr,
        "OOS_R2": res.oos_r2,
        "n_days": res.n_days,
        "n_blocks": res.n_blocks,
        "elapsed_s": res.elapsed_s,
        "degenerate": res.degenerate,
    }, ensure_ascii=False, indent=2))

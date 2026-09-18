#!/usr/bin/env python3
"""Zagent U盘一键运行入口 —— 体检 + Model-RSI 闭环。

用法（把 U盘目录拷到目标机后）:
    python run_model.py

可选参数:
    --blocks N       只跑前 N 块（默认 2，快速演示；全量 123 要 GPU）
    --iterations N   TPE 搜索轮数（默认 3）
    --fixed-epochs N 固定 epochs（默认 3 快速；全量 15）
    --n-seeds N      多 seed 降噪（默认 1，GPU 可设 2-3）
"""
import sys
from pathlib import Path

# 行缓冲：让小白双击 bat 时能看到实时进度，而不是最后才一次性打印
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # 让 zagent 包可被 import
PKG = HERE / "data" / "s13"             # 数据包路径

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Zagent Model-RSI 一键运行")
    ap.add_argument("--blocks", type=int, default=1, help="walk-forward 块数，默认1（约1分钟）；全量设 123")
    ap.add_argument("--iterations", type=int, default=2, help="TPE 搜索轮数，默认2")
    ap.add_argument("--fixed-epochs", type=int, default=3, help="固定训练轮数，默认3（快速）；全量设 15")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    from zagent.migrate import doctor

    print(f"[zagent] 数据包: {PKG}")
    rc = doctor(PKG)
    if rc != 0:
        sys.exit("体检未通过，请先解决上面的 FAIL")

    from zagent.model import run_alpha

    settings = {
        "iterations": args.iterations,
        "blocks": args.blocks,
        "max_train": args.max_train,
        "fixed_epochs": args.fixed_epochs,
        "n_seeds": args.n_seeds,
        "threads": args.threads,
        "isolated": False,     # U盘本地跑，无需子进程隔离
        "out_dir": "runs_alpha",
    }
    stats = run_alpha(PKG, settings, offline=True)
    print("\n[zagent] 完成：最佳 RankIC =", round(stats.best_rank_ic, 4))
    print("[zagent] 最佳配方:", stats.best_recipe)

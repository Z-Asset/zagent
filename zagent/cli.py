"""zagent 命令行入口 —— 一条命令驱动三个 RSI 算子。

用法:
    zagent data <面板.npy> <标签.npy>      # Data-RSI（数据清洗）
    zagent harness <config.json>            # Harness-RSI（进化 prompt/记忆/技能）
    zagent model <数据包路径>           # Model-RSI（进化 alpha 训练配方）
    zagent doctor <数据包路径>               # 体检数据+依赖

harness 默认离线零依赖；model 默认离线 TPE 搜索；都可 --online 换 DeepSeek LLM。
"""
import argparse
import sys

from .harness import run_loop
from .model import run_alpha
from .migrate import doctor
from .data import preprocess_panel, PreprocessConfig


def _data_cmd(args) -> int:
    import numpy as np
    X = np.load(args.x)
    out = preprocess_panel(X, PreprocessConfig(), verbose=True)
    np.save(args.out, out)
    print(f"清洗完成 -> {args.out}  形状 {out.shape}")
    return 0


def _harness_cmd(args) -> int:
    run_loop(args.config, offline=not args.online)
    return 0


def _model_cmd(args) -> int:
    settings = {
        "iterations": args.iterations,
        "blocks": args.blocks,
        "max_train": args.max_train,
        "fixed_epochs": args.fixed_epochs,
        "threads": args.threads,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "patience": args.patience,
        "timeout_s": args.timeout_s,
        "out_dir": args.out_dir,
        "isolated": not args.no_isolation,
    }
    run_alpha(args.pkg, settings, offline=not args.online)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="zagent",
        description="Zagent：三角色自改进闭环（Data-RSI + Harness-RSI + Model-RSI）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # data
    dp = sub.add_parser("data", help="Data-RSI：数据清洗（去极值→截面标准化→中位数填充）")
    dp.add_argument("x", help="特征面板 .npy (T,N,K)")
    dp.add_argument("out", help="清洗后输出 .npy")
    dp.set_defaults(func=_data_cmd)

    # harness
    hp = sub.add_parser("harness", help="Harness-RSI：进化系统提示词/记忆/技能")
    hp.add_argument("config", help="任务配置 JSON（如 echo.json）")
    hp.add_argument("--online", action="store_true", help="用 DeepSeek LLM 当 Refiner")
    hp.set_defaults(func=_harness_cmd)

    # model
    mp = sub.add_parser("model", help="Model-RSI：进化 alpha 训练配方")
    mp.add_argument("pkg", nargs="?", default=".", help="数据包路径（含 X/Y/train_from_xy.py）")
    mp.add_argument("--iterations", type=int, default=8)
    mp.add_argument("--blocks", type=int, default=None, help="walk-forward 块数，None=全量 123 块")
    mp.add_argument("--max-train", type=int, default=20000)
    mp.add_argument("--fixed-epochs", type=int, default=None)
    mp.add_argument("--threads", type=int, default=2)
    mp.add_argument("--seed", type=int, default=42)
    mp.add_argument("--n-seeds", type=int, default=1, help="同一配方多 seed 跑取均值降噪")
    mp.add_argument("--patience", type=int, default=3)
    mp.add_argument("--timeout-s", type=int, default=None)
    mp.add_argument("--out-dir", default="runs_alpha")
    mp.add_argument("--no-isolation", action="store_true")
    mp.add_argument("--online", action="store_true", help="用 DeepSeek LLM 当 Refiner")
    mp.set_defaults(func=_model_cmd)

    # doctor
    dp = sub.add_parser("doctor", help="体检数据包")
    dp.add_argument("pkg", nargs="?", default=".")
    dp.set_defaults(func=lambda a: doctor(a.pkg))

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

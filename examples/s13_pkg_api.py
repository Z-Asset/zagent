"""S13_csi500seq_trans 数据包对 Zagent 框架的标准接口 —— 示例模板。

框架要求数据包在**包根目录**放一个 pkg_api.py，暴露两个函数：
  read_data(pkg) -> (X, R, dates, codes)
  build_model(recipe, K, L, seed) -> fit/predict 对象

本文件是 S13 的适配器示例：数据读取、预处理、模型构造全部复用
train_from_xy.py 的成熟实现，只做签名对齐，不重写任何逻辑。

使用：把本文件复制为 S13 包根下的 pkg_api.py 即可（本模板不写进 S13）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# 模块级缓存：build_model 不接收 pkg 参数（框架接口固定），
# 所以 read_data 先被调用时把 pkg 缓存在这里，build_model 复用。
# 框架总是先 read_data 再 build_model，顺序有保证。
_CURRENT_PKG: Path | None = None


def _load_train(pkg: Path):
    """动态加载 train_from_xy.py（数据包自带的训练入口）。"""
    entry = pkg / "train_from_xy.py"
    spec = importlib.util.spec_from_file_location("pkg_train_from_xy", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pkg_train_from_xy"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_data(pkg: Path):
    """读数据 + 用 **zagent 框架正确口径** 预处理，返回 (X, R, dates, codes)。

    清洗顺序（业界标准，逐日截面）：去极值 → 截面标准化 → 中位数填充。
    用 zagent.data.preprocess 的正确实现，**不复用** train_from_xy 里的旧口径
    （旧口径是「填→截→标」，顺序错误）。

    X : (T, N, K) float32，已预处理
    R : (T, N) float32 次日收益，NaN 保留
    """
    global _CURRENT_PKG
    _CURRENT_PKG = pkg
    from zagent.data.preprocess import preprocess_panel, PreprocessConfig, build_labels

    X_raw = np.load(pkg / "X" / "X.npy").astype(float)
    R_raw = np.load(pkg / "Y" / "Y_target.npy").astype(float)
    dates = np.array([d.strip() for d in
                      (pkg / "X" / "dates.txt").read_text(encoding="utf-8").splitlines()
                      if d.strip()], dtype="datetime64[ns]")
    codes = [c.strip() for c in
             (pkg / "X" / "codes.txt").read_text(encoding="utf-8").splitlines()
             if c.strip()]
    X = preprocess_panel(X_raw, PreprocessConfig(), verbose=False)
    R = build_labels(R_raw, verbose=False)
    return X, R, dates, codes


def build_model(recipe: dict, K: int, L: int, seed: int):
    """按 recipe 构造模型，返回有 fit(X_seq, y)/predict(X_seq) 的对象。

    recipe 是框架配方 dict（含 d_model/nhead/layers/ffn/dropout/epochs/lr/batch）。
    这里做签名转换，转成数据包 build_model(key, params, seed, K, L, device)。
    """
    import torch

    if _CURRENT_PKG is None:
        raise RuntimeError("build_model 必须在 read_data 之后调用")
    mod = _load_train(_CURRENT_PKG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 复用数据包自带的 build_model，key 按数据包实际模型（S13 是 trans）
    return mod.build_model("trans", dict(recipe), seed=seed, K=K, L=L, device=device)


if __name__ == "__main__":
    pkg = Path(".")
    X, R, dates, codes = read_data(pkg)
    print(f"read_data ok: X{X.shape} R{R.shape} {len(dates)}天 {len(codes)}只")

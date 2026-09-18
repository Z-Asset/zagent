"""Data-RSI 算子 —— 资产定价数据清洗（逐日截面预处理）。

口径（顺序不可调换，每个交易日 t 独立、每个特征 k 独立）：
      1) 去极值（clip）           —— 3MAD 或分位数，把极端值拉回边界
      2) 截面驻日标准化（zscore） —— 减截面均值除截面标准差
      3) 截面缺失值中位数填充    —— 最后填，填标准化后截面的中位数

关键点：
  - 顺序严格：clip → zscore → fill。clip 在标准化前截原始值；缺失值在
    clip/zscore 阶段一直保持 NaN，最后一步用当日截面中位数填。
  - eps 统一：全模块共用 EPS，做标准差/scale 的退化判据，避免除零。
  - 缺失值用中位数（稳健），绝不用均值（被极端值拉偏）。
  - 绝不用跨期信息：每一步统计量只由当日截面决定。

本模块自包含，不 import 任何具体数据包。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 统一 eps：数值稳定性判据（标准差/scale 退化、除零保护）。
EPS = 1e-12


@dataclass
class PreprocessConfig:
    """预处理配置（逐日截面口径）。"""
    # 去极值（clip）
    winsor: str = "mad"          # "mad" | "quantile" | "none"
    mad_k: float = 3.0           # MAD 倍数（winsor="mad"）
    quantile: float = 0.01       # 双侧分位（winsor="quantile"）

    # 标准化
    standardize: str = "zscore"  # "zscore" | "rank" | "none"

    # 缺失值
    fill_method: str = "median"  # "median" | "mean" | "zero"
    min_stocks_per_day: int = 30  # 当日有效股票少于该数则整日丢弃
    drop_label_nan: bool = True   # 标签缺失的样本直接剔除


def _clip_col(col: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """去极值（clip）—— 对原始值把超出边界的值拉回边界，标准化**之前**做。

    3MAD：med ± k·MAD，MAD = median(|x - median(x)|)，1.4826 是正态下 MAD→σ
    的一致性系数。厚尾分布下 MAD 比标准差稳健得多。

    缺失值（NaN）保持不动，留给最后的填充步骤。
    """
    if cfg.winsor == "none":
        return col
    ok = np.isfinite(col)
    if ok.sum() < 3:
        return col
    x = col[ok]

    if cfg.winsor == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4826 * mad
        if scale < EPS:
            return col
        lo, hi = med - cfg.mad_k * scale, med + cfg.mad_k * scale
    elif cfg.winsor == "quantile":
        lo, hi = np.quantile(x, cfg.quantile), np.quantile(x, 1 - cfg.quantile)
    else:
        return col

    out = col.copy()
    out[ok] = np.clip(x, lo, hi)
    return out


def _standardize_col(col: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """截面标准化。zscore 减截面均值除截面标准差（ddof=0）。

    缺失值保持 NaN；退化（sd < EPS）时置 0。此处**不 clip**——去极值已在前面做。
    """
    if cfg.standardize == "none":
        return col
    ok = np.isfinite(col)
    if ok.sum() < 3:
        return col

    if cfg.standardize == "zscore":
        x = col[ok]
        mu, sd = x.mean(), x.std()
        out = col.copy()
        if sd < EPS:
            out[ok] = 0.0
        else:
            out[ok] = (x - mu) / sd
        return out

    if cfg.standardize == "rank":
        out = np.full_like(col, np.nan)
        x = col[ok]
        order = x.argsort().argsort()
        n = len(x)
        out[ok] = 2.0 * order / max(n - 1, 1) - 1.0   # → [-1, 1]
        return out

    return col


def _fill_col(col: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """截面缺失值中位数填充 —— 最后一步，填标准化后截面的中位数。

    只用当日截面统计量，不引跨期信息。
    """
    nan = ~np.isfinite(col)
    if not nan.any():
        return col
    ok = ~nan
    if ok.sum() == 0:
        return np.zeros_like(col)      # 整列缺失 → 置 0（截面中心）
    x = col[ok]
    if cfg.fill_method == "median":
        v = float(np.median(x))
    elif cfg.fill_method == "mean":
        v = float(x.mean())
    else:
        v = 0.0
    out = col.copy()
    out[nan] = v
    return out


def preprocess_panel(X_raw: np.ndarray, cfg: PreprocessConfig | None = None,
                     verbose: bool = False) -> np.ndarray:
    """对整块面板 (T, N, K) 做逐日截面预处理。

    处理顺序（每个交易日独立，顺序不可调换）：
        for t in 1..T:
            for k in 1..K:
                1) 去极值（clip）
                2) 截面驻日标准化（zscore）
                3) 截面缺失值中位数填充
            若当日有效股票 < min_stocks_per_day，整日置 NaN
    """
    cfg = cfg or PreprocessConfig()
    X = np.asarray(X_raw, dtype=float).copy()
    T, N, K = X.shape

    n_filled = 0
    n_dropped_days = 0

    for t in range(T):
        day = X[t]
        n_filled += int((~np.isfinite(day)).sum())

        valid = np.isfinite(day).any(axis=1)
        if valid.sum() < cfg.min_stocks_per_day:
            X[t] = np.nan
            n_dropped_days += 1
            continue

        for k in range(K):
            col = day[:, k]
            col = _clip_col(col, cfg)          # 1) 去极值
            col = _standardize_col(col, cfg)   # 2) 截面标准化
            col = _fill_col(col, cfg)          # 3) 缺失值中位数填充
            day[:, k] = col
        X[t] = day

    if verbose:
        print(f"  [数据清洗] 填充 {n_filled} 个缺失值；"
              f"丢弃 {n_dropped_days} 个不完整交易日（阈值 {cfg.min_stocks_per_day} 只）")
        finite = np.isfinite(X)
        print(f"  [数据清洗] 处理后：均值={np.nanmean(X):+.4f}  "
              f"标准差={np.nanstd(X):.4f}  缺失率={1-finite.mean():.4%}")

    return X


def build_labels(R_raw: np.ndarray, verbose: bool = False) -> np.ndarray:
    """处理标签：**只保留 NaN，绝不填充**。

    标签缺失绝不能填：填出来的"收益"是凭空捏造的信号。训练时含 NaN 标签的
    样本由下游剔除。
    """
    R = np.asarray(R_raw, dtype=float).copy()
    if verbose:
        n_nan = int(np.isnan(R).sum())
        print(f"  [标签] 缺失 {n_nan} 个（{n_nan/R.size:.2%}）——保留为 NaN，训练时剔除")
    return R

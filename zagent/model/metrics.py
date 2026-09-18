"""Model-RSI 框架内置指标 —— 逐日横截面 RankIC / ICIR / R² / Sharpe。

口径与 S13 复刻（exp_seq/metrics.py）逐字一致，但**不依赖任何具体数据包**，
是本框架自己的标准实现。主指标 OOS_RankIC（纯信号质量），组合层 OOS_SR 用
score 加权口径。

score 加权 = w ∝ max(score, 0)，与项目 8 篇对比表口径一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS = 252
RISK_FREE = 0.0


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, bench: np.ndarray) -> float:
    """样本外 R²，基准为各块训练期均值（论文口径）。"""
    if len(y_true) == 0:
        return np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - bench) ** 2))
    return 1.0 - sse / sst if sst > 1e-18 else np.nan


def rank_ic_series(pred: pd.DataFrame, R: pd.DataFrame,
                   min_stocks: int = 10) -> pd.Series:
    """逐日横截面 Rank IC（Spearman 秩相关）。"""
    ics: dict = {}
    common = pred.index.intersection(R.index)
    for dt in common:
        p = pred.loc[dt].values.astype(float)
        r = R.loc[dt].values.astype(float)
        m = np.isfinite(p) & np.isfinite(r)
        if m.sum() < min_stocks:
            continue
        if np.allclose(p[m], p[m][0]):
            continue
        ic, _ = spearmanr(p[m], r[m])
        if np.isfinite(ic):
            ics[dt] = float(ic)
    return pd.Series(ics)


def icir(ic: pd.Series) -> float:
    """ICIR = IC 均值 / IC 标准差。"""
    if len(ic) < 2 or ic.std() == 0:
        return np.nan
    return float(ic.mean() / ic.std())


def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def portfolio_returns(pred: pd.DataFrame, R: pd.DataFrame,
                      mode: str = "score_w",
                      top_frac: float | None = None) -> pd.Series:
    """按预测值构造组合，返回逐日收益序列。"""
    dates = pred.index.intersection(R.index)
    out: dict = {}

    for d in dates:
        s = pred.loc[d].astype(float)
        r = R.loc[d].astype(float)
        valid = np.isfinite(s.values) & np.isfinite(r.values)
        if valid.sum() < 10:
            continue
        sv, rv = s.values[valid], r.values[valid]

        if mode == "score_w":
            w = np.clip(sv, 0.0, None)
        elif mode == "sign_ew":
            w = (sv > 0).astype(float)
        elif mode == "demean_sign":
            w = (sv - sv.mean() > 0).astype(float)
        elif mode == "quantile_ls":
            q = top_frac or 0.1
            lo, hi = np.quantile(sv, q), np.quantile(sv, 1 - q)
            w = np.where(sv >= hi, 1.0, np.where(sv <= lo, -1.0, 0.0))
        elif mode == "topk_ew":
            k = max(1, int(len(sv) * (top_frac or 0.2)))
            idx = np.argsort(-sv)[:k]
            w = np.zeros_like(sv)
            w[idx] = 1.0
        else:
            raise ValueError(f"unknown portfolio mode: {mode}")

        denom = np.abs(w).sum()
        out[d] = float((w * rv).sum() / denom) if denom > 0 else 0.0

    return pd.Series(out).sort_index()


def ann_sharpe(ret: pd.Series) -> float:
    r = ret.dropna().values
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def summarize(pred: pd.DataFrame, R: pd.DataFrame, bench: pd.Series,
              name: str = "", pred_train: pd.DataFrame | None = None) -> dict:
    """一次算齐 OOS 指标。返回 dict（OOS_RankIC / ICIR / OOS_SR / OOS_R2 / n_days 等）。"""
    oos_idx = pred.dropna(how="all").index
    if len(oos_idx) == 0:
        return {"模型": name}

    # 常数预测检测
    n_const = 0
    for d in oos_idx:
        s = pred.loc[d]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        s = s.dropna()
        if len(s) > 1 and len(s.unique()) <= 1:
            n_const += 1
    degenerate = n_const > 0.5 * len(oos_idx)

    pv = pred.loc[oos_idx].values.ravel().astype(float)
    rv = R.loc[oos_idx].values.ravel().astype(float)
    m = np.isfinite(pv) & np.isfinite(rv)
    bv = np.repeat(bench.loc[oos_idx].values.astype(float), pred.shape[1])[m]

    ic = rank_ic_series(pred.loc[oos_idx], R.loc[oos_idx])
    ret = portfolio_returns(pred.loc[oos_idx], R.loc[oos_idx], mode="score_w")

    out = {
        "模型": name,
        "坍缩": "是" if degenerate else "",
        "OOS_SR": ann_sharpe(ret),
        "OOS_RankIC": float(ic.mean()) if len(ic) else np.nan,
        "OOS_ICIR": icir(ic),
        "OOS_R2": r2_oos(rv[m], pv[m], bv),
        "OOS_命中率": hit_ratio(rv[m], pv[m]),
        "OOS_年化收益": float(ret.mean() * TRADING_DAYS) if len(ret) else np.nan,
        "OOS_年化波动": float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else np.nan,
        "n_days": int(len(oos_idx)),
    }

    if pred_train is not None:
        itr = pred_train.dropna(how="all").index.intersection(R.index)
        if len(itr) > 0:
            iic = rank_ic_series(pred_train.loc[itr], R.loc[itr])
            out["IS_RankIC"] = float(iic.mean()) if len(iic) else np.nan
            out["IS_ICIR"] = icir(iic)
            out["n_days_is"] = int(len(itr))

    return out

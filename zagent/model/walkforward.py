"""Model-RSI 框架内置 walk-forward 协议 —— 不依赖任何数据包。

数据包只提供两个最小接口：
  - read_data(pkg) -> (X, R, dates, codes)  面板张量 + 标签 + 交易日 + 股票代码
  - build_model(recipe, K, L, seed) -> 有 fit(X_seq, y)/predict(X_seq) 的对象

其余全部由框架自己实现，保证口径统一、可复算：
  - expanding walk-forward 窗口切分（训练/验证/测试，**训练窗与测试块间 2 天空白隔离**）
  - 序列切窗（信号日 t 的窗口 = [t-L+1, t]，杜绝前视）
  - 每块「训练段 fit 候选 → 独立验证段选优 → 训练窗重训 → 测试块预测」
  - 多 seed：同一配方跑多个 seed，指标取均值/标准差降噪
  - 指标计算（OOS_RankIC / ICIR / OOS_SR）

关键防泄漏约束（业界口径，硬编码，不可关）：
  - GAP = 2：训练/验证窗终点 = 测试块起点 − 2，隔离样本内外
  - seq_index(t, L)：窗口结束于 t，绝不用未来特征
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import summarize

# walk-forward 窗口协议常量（业界口径，硬编码）
START = 252        # 第一个测试块起点（留一年初始训练窗）
BLOCK = 21         # 测试块长度（约一个月）
VAL = 21           # 验证段长度（选超参）
GAP = 2            # 训练/验证窗与测试块之间的空白（隔离样本内外）


@dataclass
class Window:
    wi: int
    tr_s: int        # 训练起点（含）
    tr_e: int        # 训练终点（含）
    va_s: int        # 验证起点（含）
    va_e: int        # 验证终点（含）
    te_s: int        # 测试起点（含）
    te_e: int        # 测试终点（不含）

    @property
    def n_train_days(self) -> int:
        return self.tr_e - self.tr_s + 1

    @property
    def n_test_days(self) -> int:
        return self.te_e - self.te_s


def expanding_windows(n_days: int, start: int = START, block: int = BLOCK,
                      val: int = VAL, gap: int = GAP,
                      max_blocks: int | None = None) -> list[Window]:
    """构造 expanding walk-forward 窗口列表。

    gap = 2 是**硬约束**：tr_e = te_s - gap，训练/验证窗终点在测试块起点前
    2 天，隔离样本内与样本外，防止短期收益自相关导致的泄漏。
    """
    wins: list[Window] = []
    wi = 0
    te_s = start
    while te_s < n_days - 1:
        te_e = min(te_s + block, n_days - 1)
        tr_e = te_s - gap
        if tr_e < val:
            te_s += block
            continue
        wins.append(Window(wi=wi, tr_s=0, tr_e=tr_e,
                           va_s=tr_e - val + 1, va_e=tr_e,
                           te_s=te_s, te_e=te_e))
        wi += 1
        te_s += block
        if max_blocks is not None and len(wins) >= max_blocks:
            break
    return wins


def make_sequences(X: np.ndarray, L: int) -> np.ndarray:
    """把面板 (T, N, K) 切成时间窗序列 (T-L+1, N, L, K)，零拷贝视图。

    样本下标与信号日：Xseq[i] 的窗口结束于 i+L-1；信号日 t 的窗口 = seq_index(t,L)。
    只沿时间轴滑窗（axis=0），股票/特征轴不动，杜绝前视。
    """
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(X, (L,), axis=0)      # (T-L+1, N, K, L)
    return np.transpose(sw, (0, 1, 3, 2))          # (T-L+1, N, L, K)


def seq_index(t: int, L: int) -> int:
    """信号日 t 对应序列样本下标 = t - (L-1)。"""
    return t - (L - 1)


def _make_xy(Xseq, R, t_idx, L, cap=0, seed=0):
    """把面板时间窗 + 标签拼成训练矩阵，剔除 NaN 标签样本。"""
    K = Xseq.shape[3]
    t_idx = np.asarray(t_idx)
    lo, hi = 0, Xseq.shape[0]
    if t_idx.size and (t_idx.min() < lo or t_idx.max() >= hi):
        raise IndexError(f"信号日越界: [{t_idx.min()},{t_idx.max()}] 超出 [{lo},{hi})")

    if cap and len(t_idx) * Xseq.shape[1] > cap:
        rng = np.random.default_rng(seed)
        n_take = max(1, cap // Xseq.shape[1])
        t_sel = rng.choice(t_idx, size=min(n_take, len(t_idx)), replace=False)
        t_sel.sort()
    else:
        t_sel = t_idx

    n_t, N = len(t_sel), Xseq.shape[1]
    if n_t == 0:
        return np.empty((0, L, K), np.float32), np.empty(0, np.float32)

    Xf = np.empty((n_t * N, L, K), dtype=np.float32)
    yf = np.empty(n_t * N, dtype=np.float32)
    for i, t in enumerate(t_sel):
        j = i * N
        Xf[j:j + N] = Xseq[seq_index(t, L)]
        yf[j:j + N] = R[t]
    ok = np.isfinite(yf)
    if ok.all():
        return Xf, yf
    return Xf[ok], yf[ok]


def run_walk_forward(X: np.ndarray, R: np.ndarray, dates, codes,
                     build_model, L: int, recipe: dict, *,
                     seed: int = 42, max_train: int = 20000,
                     max_blocks: int | None = None, verbose: bool = False,
                     n_seeds: int = 1) -> dict:
    """内置 expanding walk-forward。

    参数
    ----
    build_model(recipe, K, L, seed) -> 模型对象（有 fit/predict）
    n_seeds : 同一配方跑多个 seed，指标取均值/标准差，降噪（随机数多设置）。

    返回
    ----
    {"metrics": {...均值指标...}, "per_seed": [...], "n_blocks": int}
    """
    import time
    t0 = time.time()

    n_days = X.shape[0]
    wins = expanding_windows(n_days, max_blocks=max_blocks)
    Xseq = make_sequences(X, L)
    K = X.shape[2]

    Rdf = pd.DataFrame(R, index=pd.DatetimeIndex(dates), columns=list(codes))

    per_seed = []
    for s_i in range(n_seeds):
        seed_i = seed + s_i
        pred = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=list(codes), dtype=float)
        bench = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)

        for w in wins:
            tr_t = np.arange(max(L - 1, w.tr_s), w.tr_e)
            va_t = np.arange(w.va_s, w.va_e + 1)
            te_t = np.arange(w.te_s, w.te_e)

            Xtr, ytr = _make_xy(Xseq, R, tr_t, L, cap=max_train, seed=seed_i + w.wi)
            Xva, yva = _make_xy(Xseq, R, va_t, L, cap=50000, seed=seed_i + w.wi)
            if len(ytr) < 200 or len(yva) < 50:
                continue

            # 训练段 fit 候选（单配方：这里只训一个，等价于 recipe 本身就是候选）
            rng = np.random.default_rng(seed_i + w.wi)
            if max_train and len(ytr) > max_train:
                idx = rng.choice(len(ytr), max_train, replace=False)
                Xtr_s, ytr_s = Xtr[idx], ytr[idx]
            else:
                Xtr_s, ytr_s = Xtr, ytr

            model = build_model(recipe, K, L, seed_i)
            model.fit(Xtr_s, ytr_s)

            # 测试块预测
            for t in te_t:
                x = Xseq[seq_index(t, L)]
                ok = np.isfinite(R[t])
                if ok.any():
                    pred.iloc[t, np.where(ok)[0]] = model.predict(x[ok])
                    bench.iloc[t] = float(ytr.mean())

        row = summarize(pred, Rdf, bench, name=f"seed{seed_i}")
        per_seed.append({"seed": seed_i, "metrics": row})

    # 聚合多 seed：主指标取均值
    agg = {}
    for key in ("OOS_SR", "OOS_RankIC", "OOS_ICIR", "OOS_R2", "OOS_命中率"):
        vals = [s["metrics"].get(key) for s in per_seed if s["metrics"].get(key) is not None]
        if vals:
            agg[key] = float(np.nanmean(vals))
            agg[key + "_std"] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        else:
            agg[key] = np.nan
    agg["n_days"] = per_seed[0]["metrics"].get("n_days", 0) if per_seed else 0
    agg["elapsed_s"] = round(time.time() - t0, 1)

    return {"metrics": agg, "per_seed": per_seed, "n_blocks": len(wins)}

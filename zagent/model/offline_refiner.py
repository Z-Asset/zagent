"""离线智能 Refiner —— TPE（Tree-structured Parzen Estimator）贝叶斯优化。

离线版不靠 LLM，而是用经典贝叶斯超参优化。TPE 是 Optuna 的默认算法，
在离散超参空间上被反复证明优于随机/网格搜索。它把"智能"放在五处：

  1. 历史记忆 + 去重 —— 已评估的配方绝不重提，不浪费评估预算；
  2. 分位数建模 —— 把历史按指标分成"好组 l(x)"和"差组 g(x)"，对每个维度
     用频率表（离散空间的核密度）估计某点属于好组的概率比 l(x)/g(x)；
  3. 采集函数 —— 选 l(x)/g(x) 最大的、尚未见过的点，天然平衡探索与利用；
  4. 初始覆盖 —— 前 n_init 轮用确定性拉丁超立方铺满空间，冷启动不盲猜；
  5. 早停 —— 由主循环配合：连续 patience 轮无改进即停。

为什么"不输在线版"：LLM 当超参优化器会重复、幻觉、越界（输出 d_model=100
之类非法值），且对噪声敏感。TPE 无此问题；blocks=1 的 RankIC 噪声大，TPE 的
分位 + 平滑对它更鲁棒。

本模块是纯搜索算法，不 import 任何 alpha 训练代码；空间通过构造函数注入，
可复用于任意离散超参搜索。可单独用合成历史做单元测试，秒级验证"确实更智能"。
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable


class OfflineTPERefiner:
    """离散空间上的 TPE 优化器，充当离线 Refiner。"""

    def __init__(
        self,
        space: dict[str, list],
        dim_order: list[str],
        *,
        seed: int = 42,
        n_init: int = 4,
        gamma: float = 0.3,
        alpha: float = 1.0,
        fixed: dict | None = None,
    ):
        self.space = space
        self.fixed = dict(fixed or {})
        # 只搜索「可调」维度；fixed 里的维度（如 epochs）从建模中剔除，
        # 避免出现「搜一个维度但评估时不生效」的假维度。
        self.dim_order = [d for d in dim_order if d in space and d not in self.fixed]
        self.n_init = n_init
        self.gamma = gamma
        self.alpha = alpha          # 拉普拉斯平滑系数
        self.seen: set[tuple] = set()
        self._init_count = 0
        self._candidates = self._enumerate_valid()
        self._primes = [3, 5, 7, 11, 13, 17, 19, 23]

    def _full(self, partial: dict) -> dict:
        """把搜索维度上的部分配方补全为完整配方（并入 fixed）。"""
        return {**partial, **self.fixed}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def propose(self, history: list[dict]) -> dict:
        """给定历史，提议下一个完整配方（含 fixed 维度）。历史项须含 recipe + rank_ic。"""
        entries = self._collect(history)
        for recipe, _ in entries:
            self.seen.add(self._key(recipe))

        if self._init_count < self.n_init:
            candidate = self._latin(self._init_count)
            self._init_count += 1
            candidate = self._skip_seen(candidate)
        else:
            candidate = self._tpe(entries)
        full = self._full(candidate)
        self.seen.add(self._key(full))
        return dict(full)

    def explain(self, recipe: dict, best: dict | None) -> str:
        """给选点一个可读理由（模仿 LLM 的 reason 字段，便于审计）。"""
        if best is None:
            return "初始覆盖采样"
        diffs = []
        for dim in self.dim_order:
            if recipe.get(dim) != best.get(dim):
                diffs.append(f"{dim} {best.get(dim)}->{recipe.get(dim)}")
        return "调整 " + ", ".join(diffs) if diffs else "在最优附近保持、局部探索"

    # ------------------------------------------------------------------
    # 历史收集与建模
    # ------------------------------------------------------------------
    @staticmethod
    def _key(recipe: dict) -> tuple:
        return tuple(recipe.get(d) for d in sorted(recipe))

    def _collect(self, history: list[dict]) -> list[tuple[dict, float]]:
        out: list[tuple[dict, float]] = []
        for h in history:
            r = h.get("recipe")
            m = h.get("rank_ic")
            if not isinstance(r, dict) or m is None:
                continue
            try:
                m = float(m)
            except (TypeError, ValueError):
                continue
            if m != m:              # NaN
                m = float("-inf")
            out.append((r, m))
        out.sort(key=lambda x: x[1], reverse=True)   # 指标降序
        return out

    def _freq(self, entries: list[tuple[dict, float]]) -> dict[str, tuple[list[float], float]]:
        """每个维度返回 (按合法值顺序的平滑计数, 计数总和)。"""
        freqs: dict[str, tuple[list[float], float]] = {}
        for dim in self.dim_order:
            vals = self.space[dim]
            idx = {v: i for i, v in enumerate(vals)}
            counts = [self.alpha] * len(vals)
            for recipe, _ in entries:
                if dim in recipe:
                    v = recipe[dim]
                    if v in idx:
                        counts[idx[v]] += 1.0
            total = sum(counts)
            freqs[dim] = (counts, total)
        return freqs

    def _score(self, recipe: dict,
               l: dict[str, tuple[list[float], float]],
               g: dict[str, tuple[list[float], float]]) -> float:
        """TPE 采集值 = log( l(x)/g(x) )，维度独立假设下逐维求和。"""
        s = 0.0
        for dim in self.dim_order:
            vals = self.space[dim]
            v = recipe.get(dim)
            if v not in vals:
                return float("-inf")
            i = vals.index(v)
            l_counts, l_total = l[dim]
            g_counts, g_total = g[dim]
            l_p = l_counts[i] / l_total
            g_p = g_counts[i] / g_total
            if g_p <= 0:
                return float("inf")
            s += math.log(l_p / g_p)
        return s

    # ------------------------------------------------------------------
    # 候选生成
    # ------------------------------------------------------------------
    def _enumerate_valid(self) -> list[dict]:
        """枚举全部合法配方（d_model % nhead == 0 约束内嵌）。"""
        dims = self.dim_order
        spaces = [self.space[d] for d in dims]
        out: list[dict] = []
        for combo in itertools.product(*spaces):
            recipe = dict(zip(dims, combo))
            if not self._valid(recipe):
                continue
            out.append(recipe)
        return out

    def _valid(self, recipe: dict) -> bool:
        if "d_model" in recipe and "nhead" in recipe:
            if recipe["nhead"] != 0 and recipe["d_model"] % recipe["nhead"] != 0:
                return False
        return True

    def _latin(self, k: int) -> dict:
        """确定性拉丁超立方：第 k 个初始点，各维取不同偏移，保证铺开。"""
        recipe: dict = {}
        d_model = self._pick("d_model", k, 3)
        recipe["d_model"] = d_model
        # nhead 必须整除 d_model
        valid_heads = [h for h in self.space.get("nhead", []) if d_model % h == 0]
        recipe["nhead"] = valid_heads[k % len(valid_heads)] if valid_heads else d_model
        for i, dim in enumerate(self.dim_order):
            if dim in ("d_model", "nhead"):
                continue
            recipe[dim] = self._pick(dim, k, self._primes[i % len(self._primes)])
        return recipe

    def _pick(self, dim: str, k: int, stride: int):
        vals = self.space[dim]
        return vals[(k * stride) % len(vals)]

    def _skip_seen(self, recipe: dict) -> dict:
        """若拉丁点已被评估过，沿候选列表找第一个未见过的。"""
        if self._key(recipe) not in self.seen:
            return recipe
        for cand in self._candidates:
            if self._key(cand) not in self.seen:
                return cand
        return recipe   # 空间耗尽（几乎不可能）

    def _tpe(self, entries: list[tuple[dict, float]]) -> dict:
        if not entries:
            return self._latin(self._init_count)
        k_top = max(2, int(len(entries) * self.gamma))
        top = entries[:k_top]
        rest = entries[k_top:]
        l = self._freq(top)
        g = self._freq(rest) if rest else l

        best_score = float("-inf")
        best_cand = None
        for cand in self._candidates:
            if self._key(cand) in self.seen:
                continue
            s = self._score(cand, l, g)
            if s > best_score:
                best_score, best_cand = s, cand
        if best_cand is None:
            return self._skip_seen(self._latin(self._init_count))
        return best_cand


if __name__ == "__main__":
    # 自检：合成历史证明 TPE 会向高指标区域收敛。
    from zagent.model.alpha_eval import RECIPE_SPACE, REQUIRED_KEYS  # noqa: 仅演示用

    tpe = OfflineTPERefiner(RECIPE_SPACE, REQUIRED_KEYS, seed=0, n_init=2)
    # 造历史：d_model=96 的点 RankIC 明显高，其余低。
    fake = []
    for i in range(6):
        if i < 3:
            fake.append({"recipe": {"d_model": 96, "nhead": 4, "layers": 1, "ffn": 128,
                                    "dropout": 0.2, "epochs": 3, "lr": 0.001, "batch": 4096},
                         "rank_ic": 0.12 + i * 0.001})
        else:
            fake.append({"recipe": {"d_model": 32, "nhead": 4, "layers": 1, "ffn": 128,
                                    "dropout": 0.1, "epochs": 3, "lr": 0.001, "batch": 4096},
                         "rank_ic": 0.05})
    print("candidate pool size:", len(tpe._candidates))
    for _ in range(3):
        p = tpe.propose(fake)
        print("propose:", p["d_model"], p["nhead"], p["layers"], p["lr"])

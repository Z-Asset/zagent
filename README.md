# Zagent — 统一 RSI 三角色自改进闭环

一个框架，三个算子。参照 MetaRSI-v1（Data-RSI + Harness-RSI + Model-RSI）与 Continual Harness 三格骨架。

## 三个算子

| 算子 | 进化对象 | 入口 |
|---|---|---|
| **Data-RSI** | 数据清洗 + 经验提取 | `zagent data <X.npy> <out.npy>` |
| **Harness-RSI** | 系统提示词 / 记忆 / 技能（不动权重） | `zagent harness <config>` |
| **Model-RSI** | alpha 训练配方（d_model/nhead/lr/…） | `zagent model <数据包>` |

三者共享 `ChatModel`（OpenAI 兼容客户端，DeepSeek/Qwen/GLM/GPT 通吃）。

## 三角色

| 角色 | 职责 |
|---|---|
| Operator 运行模型 | 跑任务 / 训练，产出轨迹 |
| Evaluator 验收模型 | 封闭判分（RankIC / 判对错） |
| Refiner 改进模型 | 读信号，提议改动（TPE / LLM） |

循环从不重置：运行 → 验收 → 诊断 → 提议 → 回放验证 → 只保留严格更优 → 继续。

## 一条命令

```bash
pip install -e .                          # 装包（提供 zagent 命令）
zagent data X.npy X_clean.npy             # Data-RSI：数据清洗
zagent harness config/echo.json           # Harness-RSI 闭环（离线零依赖）
zagent model <数据包路径>                 # Model-RSI 闭环（离线 TPE）
zagent doctor <数据包路径>                # 体检
```

默认离线；`--online` 换 DeepSeek LLM 当 Refiner。GPU 自动走 CUDA。

## Data-RSI 清洗口径

逐日截面，顺序不可调换：**去极值(clip) → 截面驻日标准化(zscore) → 截面缺失值中位数填充**。eps 统一。绝不用跨期信息。

## Model-RSI：框架内置 walk-forward，不依赖数据包

框架**自己实现** expanding walk-forward 协议，数据包只暴露两个接口（`pkg_api.py`）：

```python
def read_data(pkg):    # -> (X, R, dates, codes)  已预处理的 (T,N,K) + (T,N)
def build_model(recipe, K, L, seed):  # -> 有 fit/predict 的对象
```

框架保证的口径（硬编码，不可关）：
- **训练/验证窗与测试块之间 2 天空白隔离**（GAP=2，防短期收益自相关泄漏）
- 序列切窗 `seq_index(t, L)`：窗口结束于 t，绝不用未来特征
- 多 seed：同一配方跑多个 seed，指标取均值/标准差降噪（`--n-seeds`）
- 指标口径：OOS_RankIC / ICIR / OOS_SR（score 加权）

示例适配器见 `examples/s13_pkg_api.py`。

## 目录结构

```
Zagent/
  zagent/
    __init__.py         统一入口，导出三算子
    models.py           共享 ChatModel（含离线 Mock/Lambda）
    cli.py              统一 CLI（data/harness/model/doctor）
    migrate.py          体检（sha1 对账 + 依赖 + 数据 + CUDA）
    data/               Data-RSI：preprocess.py（清洗）+ extract.py（经验提取）
    harness/            Harness-RSI 算子
    model/              Model-RSI 算子：metrics.py + walkforward.py + alpha_eval.py + alpha_rsi.py + offline_refiner.py
  examples/             s13_pkg_api.py（数据包接口示例）
  src/index.ts          DSH 插件壳
  plugins/zagent/       SKILL.md
  config/               示例配置
  docs/                 迁移 + GPU 指南
```

## 关键约束

- `fixed_epochs` 必须设（Model-RSI），否则 TPE 搜的 epochs 维度不生效。
- 不改训练函数体；配方过 `validate_recipe` 合法空间校验。
- Evaluator 封闭；数据包不进代码仓库。

## 相关项目

- `alpharsi`（独立项目）：仅 Model-RSI 算子，本框架的前身之一。
- `attest`（独立项目）：训练鉴证，与本框架无关。

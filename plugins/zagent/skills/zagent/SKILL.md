---
name: zagent
description: >
  Zagent：统一 RSI 三角色自改进闭环框架。三个算子——Data-RSI（数据清洗：去极值→截面驻日标准化→中位数填充 +
  经验提取）、Harness-RSI（进化系统提示词/记忆/技能）、Model-RSI（进化 alpha 序列模型的训练配方，
  框架内置 walk-forward）。Operator 跑任务、Evaluator 封闭判分、Refiner 用 TPE 贝叶斯搜索（离线）
  或 DeepSeek LLM（在线）提议改动，只保留严格更优。离线一条命令跑，GPU 自动走 CUDA。
  当哥说"进化这个模型"、"改这个 harness"、"清洗这份数据"、"跑 zagent"、"给这个 alpha 包扫超参"、
  "把这个框架迁移到别的机器"时触发。
---

# Zagent

统一 RSI 三角色闭环，一个框架三个算子。**不要凭记忆复述参数**——先看
`docs/`，命令行的东西以 `zagent --help` 为准。

## 三个算子

| 算子 | 进化对象 | 入口 |
|---|---|---|
| Data-RSI | 数据清洗 + 经验提取 | `zagent data <X.npy> <out.npy>` |
| Harness-RSI | 系统提示词 / 记忆 / 技能（不动权重） | `zagent harness <config>` |
| Model-RSI | alpha 训练配方（d_model/nhead/lr/…） | `zagent model <数据包>` |

三角色：Operator（运行模型）、Evaluator（封闭验收）、Refiner（提议改动）。

## 一条命令

```
zagent data <X.npy> <out.npy>        # Data-RSI：数据清洗
zagent harness <任务配置.json>        # Harness-RSI 闭环
zagent model <数据包路径>             # Model-RSI 闭环（离线 TPE）
zagent doctor <数据包路径>            # 体检：sha1 对账 + 依赖 + 数据 + CUDA
```

默认离线零依赖；`--online` 换 DeepSeek LLM 当 Refiner。

## Data-RSI 清洗口径

逐日截面，顺序不可调换：**去极值(clip) → 截面驻日标准化(zscore) → 截面缺失值中位数填充**。eps 统一，绝不用跨期信息。

## Model-RSI：框架内置 walk-forward

框架自己实现 walk-forward，数据包只暴露 `pkg_api.py` 的 `read_data` + `build_model` 两个接口。框架硬编码保证：2 天空白隔离（GAP=2）、序列切窗无前视、多 seed 降噪。

## 成本约束（Model-RSI 必读）

CPU 上真实评估单点约 345 秒，全量 123 块不可行。**真正出结论在 GPU 上跑全量**
（`--blocks` 不传 = 全量 + `--fixed-epochs 15`）。`fixed_epochs` 必须设。

## 硬边界

- 只改数据 / harness / 训练配方，不动模型权重、不改训练函数体。
- Evaluator 封闭：Refiner 看不到、改不了评判器和测试集。
- 配方必须过 `validate_recipe` 合法空间校验，越界拒绝。
- 数据包（X/Y 张量）不进代码仓库，用 migrate 打包 + sha1 对账分发。

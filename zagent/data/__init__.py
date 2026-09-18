"""Data-RSI 算子 —— 数据状态 D 的改进：清洗 + 经验提取 + 验证。

职责（对应 MetaRSI-v1 的 Data-RSI）：
  - 数据清洗（preprocess）：逐日截面「去极值 → 截面驻日标准化 → 截面缺失值中位数填充」
  - 经验提取（extract）：从执行轨迹提取经过验证的训练记录，标记能力边界
  - 验证（validate）：生成者-验证者隔离校验，只有一致才保留记录

Data-RSI 不改模型行为、不改 harness，只产出数据工件，供 Model-RSI / Harness-RSI 使用。
"""
from .preprocess import (
    PreprocessConfig, preprocess_panel, build_labels, EPS,
)
from .extract import (
    extract_experience, annotate_fail_causes, ExperienceRecord, FailCause,
)

__all__ = [
    "PreprocessConfig", "preprocess_panel", "build_labels", "EPS",
    "extract_experience", "annotate_fail_causes", "ExperienceRecord", "FailCause",
]

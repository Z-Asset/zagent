"""Zagent —— 统一 RSI 三角色自改进闭环框架。

一个框架，三个算子（参照 MetaRSI-v1）：
  - Data-RSI（zagent.data）：数据清洗 + 经验提取，产出数据工件
  - Harness-RSI（zagent.harness）：进化系统提示词 / 记忆 / 技能，不动权重
  - Model-RSI（zagent.model）：进化 alpha 模型训练配方，不动 prompt

三者共享 zagent.models.ChatModel（OpenAI 兼容客户端）。

用法：
  zagent harness <任务配置>       # Harness-RSI 闭环
  zagent model <数据包路径>         # Model-RSI 闭环
  zagent doctor <数据包路径>       # 体检
"""
from .models import ChatModel, MockModel, LambdaModel
from . import data
from . import harness
from . import model

__all__ = [
    "ChatModel", "MockModel", "LambdaModel",
    "data", "harness", "model",
]

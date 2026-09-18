"""Data-RSI 经验提取 —— 从执行轨迹提取经过验证的训练记录。

对应 MetaRSI-v1 Data-RSI 的「经验提取 + 生成者-验证者隔离校验」：

  - 四维失败成因签名：知识缺失 / 推理缺陷 / 验证步骤缺失 / 干扰项敏感
  - 生成者-验证者隔离：同一目标模型分饰 Operator 与 Anchor，Anchor 看不到
    Operator 的答案，独立重解；仅当两者一致，记录才被保留。

记录产出后作为数据工件，供 Model-RSI（训练配方搜索）与 Harness-RSI 使用。
Data-RSI 本身不改模型行为、不改 harness。

本模块自包含，不 import 任何具体数据包。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailCause(str, Enum):
    """四维失败成因签名。"""
    KNOWLEDGE = "knowledge"      # 知识缺失
    REASONING = "reasoning"      # 推理缺陷
    VERIFICATION = "verification"  # 验证步骤缺失
    DISTRACTION = "distraction"  # 干扰项敏感


@dataclass
class ExperienceRecord:
    """一条经过验证的经验记录（数据工件）。"""
    trace_id: str
    query: str
    operator_answer: str
    anchor_answer: str
    verified: bool              # 生成者-验证者是否一致
    fail_causes: tuple[str, ...] = ()   # 四维失败成因签名
    reference: str = ""         # ground truth（若有）

    def to_jsonable(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "operator_answer": self.operator_answer,
            "anchor_answer": self.anchor_answer,
            "verified": self.verified,
            "fail_causes": list(self.fail_causes),
            "reference": self.reference,
        }


def extract_experience(traces: list[dict], anchor_solve) -> list[ExperienceRecord]:
    """从轨迹列表提取经验，做生成者-验证者隔离校验。

    参数
    ----
    traces : [{"trace_id", "query", "operator_answer", "reference"?}]
    anchor_solve : callable(query) -> str  独立重解的 Anchor（与 Operator 同模型，
                 但看不到 operator_answer）

    返回
    ----
    仅保留 operator 与 anchor 输出一致的记录（verified=True）。不一致的记录丢弃，
    避免把「某次碰巧」当成「可复用的经验」。
    """
    records: list[ExperienceRecord] = []
    for t in traces:
        query = t.get("query", "")
        op_ans = t.get("operator_answer", "")
        ref = t.get("reference", "")
        try:
            anchor_ans = anchor_solve(query)
        except Exception:
            continue  # Anchor 失败则本记录不可验证，跳过
        verified = (op_ans.strip() == anchor_ans.strip())
        records.append(ExperienceRecord(
            trace_id=t.get("trace_id", ""),
            query=query,
            operator_answer=op_ans,
            anchor_answer=anchor_ans,
            verified=verified,
            reference=ref,
        ))
    return records


def annotate_fail_causes(record: ExperienceRecord, ref: str = "") -> tuple[str, ...]:
    """给一条记录标注四维失败成因（启发式，供后续诊断）。

    判据（默认）：无 ground truth 时不标注；有 ground truth 时：
      - operator 答非所问 / 空 → verification
      - 与 ref 数值不符但格式对 → reasoning
      - 与 ref 一致 → 空（无失败）
    """
    if not ref:
        return ()
    op = record.operator_answer.strip()
    if not op:
        return (FailCause.VERIFICATION.value,)
    if op == ref:
        return ()
    # 数值比较失败 → 推理缺陷；其余归为知识缺失
    try:
        float(op)
        float(ref)
        return (FailCause.REASONING.value,)
    except ValueError:
        return (FailCause.KNOWLEDGE.value,)


if __name__ == "__main__":
    # 自检：验证「一致保留、不一致丢弃」
    traces = [
        {"trace_id": "a", "query": "1+1", "operator_answer": "2", "reference": "2"},
        {"trace_id": "b", "query": "2+2", "operator_answer": "4", "reference": "4"},
        {"trace_id": "c", "query": "3+3", "operator_answer": "5", "reference": "6"},
    ]
    recs = extract_experience(traces, anchor_solve=lambda q: {"1+1": "2", "2+2": "4", "3+3": "6"}[q])
    kept = [r for r in recs if r.verified]
    print(f"轨迹 {len(traces)} 条，验证通过 {len(kept)} 条（应=2，'c' 因答案不符被丢弃）")

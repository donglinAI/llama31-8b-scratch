"""多选题（MMLU / CMMLU / C-Eval / GPQA 的选择部分）评测。

两种评测范式（必须分清）
=========================

**范式 A：loglikelihood（判别式 / "PPL 范式"）**
    对每个候选答案 ``c``，把 ``context + c`` 拼起来喂给模型，只对 ``c`` 里
    的 token 求条件概率之和：

    .. math::
        s(c) = \\sum_{t=c_1}^{c_{|c|}} \\log p_\\theta\\big(t \\mid \\text{context}, t_{<t}\\big)

    然后选 :math:`\\arg\\max_c s(c)`。它**不让模型生成**，只让模型"打分"，
    所以快、可复现、不需要 sampling，是 MMLU/CMMLU/C-Eval/MedQA 等
    benchmark 的默认范式。lm-eval-harness 里这种任务叫
    ``task_type="multiple_choice"``。

**范式 B：gen（生成式）**
    让模型真的吐出一段文本（可能是 "A"、可能是一段推理、可能是自由答案），
    再用正则/规则/另一个模型（judge）抽取并判定对错。GPQA、数学竞赛题、
    开放问答走这条。它更接近真实使用，但方差大、依赖 sampling 与解析。

本模块只实现范式 A 的**打分聚合**部分（不绑死任何模型）：
调用方负责把每个选项的 logits 算好传进来。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from llama31_8b_scratch.evaluation.metrics_core import accuracy
from llama31_8b_scratch.evaluation.perplexity import log_softmax

__all__ = [
    "sequence_logprob",
    "mc_choice_logprobs",
    "rank_choice",
    "mc_accuracy",
]


def sequence_logprob(logits: np.ndarray, continuation_ids: np.ndarray) -> float:
    """单个选项续写的总 log 概率。

    形状约定：

    - ``logits``: ``[k, V]``，第 t 行是模型在续写第 t 个 token 时的 logits
      （由调用方对齐：把 context 最后一个 token 的输出 logit 作为续写第一个
      token 的预测）。
    - ``continuation_ids``: ``[k]``，该选项续写段的真实 token id。

    Returns:
        该选项的 ``sum_t log p(t | context, 前文)``，恒为 <=0。
    """
    logits = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(continuation_ids, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是 [k, V]，收到 {logits.shape}")
    if ids.ndim != 1 or ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"continuation_ids 必须是 [k] 且与 logits 行数一致，收到 {ids.shape}"
        )
    logp = log_softmax(logits, axis=-1)  # [k, V]
    token_logp = logp[np.arange(logits.shape[0]), ids]  # [k]
    return float(token_logp.sum())


def mc_choice_logprobs(
    per_choice_logits: Sequence[np.ndarray],
    per_choice_ids: Sequence[np.ndarray],
) -> List[float]:
    """对一组候选选项逐个求 logprob 分数。

    Args:
        per_choice_logits: 长度 C 的列表，第 c 个元素形状 ``[k_c, V]``。
        per_choice_ids: 长度 C 的列表，第 c 个元素形状 ``[k_c]``。

    Returns:
        长度 C 的分数列表，越大越好。
    """
    if len(per_choice_logits) != len(per_choice_ids):
        raise ValueError("logits 与 ids 长度不一致")
    return [
        sequence_logprob(l, i)
        for l, i in zip(per_choice_logits, per_choice_ids)
    ]


def rank_choice(scores: Sequence[float]) -> int:
    """返回分数最高的选项索引（ties 取第一个）。"""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("rank_choice 收到空分数列表")
    return int(np.argmax(arr))


def mc_accuracy(records: Sequence[Dict]) -> float:
    """在一批多选题上算 accuracy。

    Args:
        records: 每条记录是一个 dict，字段：

            - ``scores``: 长度 C 的 logprob 分数列表（由调用方用模型算好）
            - ``gold``: 正确选项的整数索引
            - 其它字段（question_id 等）被忽略。

    Returns:
        准确率 [0,1]。
    """
    preds: List[int] = []
    labels: List[int] = []
    for rec in records:
        preds.append(rank_choice(rec["scores"]))
        labels.append(int(rec["gold"]))
    return accuracy(preds, labels)

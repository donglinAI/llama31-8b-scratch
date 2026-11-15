"""手写困惑度（Perplexity, PPL）：交叉熵 → NLL → PPL 的无跳步推导。

为什么需要 PPL
==============
语言模型在一批文本上"好不好"，最直接的度量是：把这段文本喂进去，模型对每个
真实出现的词打了多低的概率。PPL 就是这件事的指数化版本，量纲是"模型在每个
位置平均要在多少个候选词里犹豫"——PPL=100 直观等价于模型平均"猜 100 次才
猜中一个词"。

数学链路（必须记住这三步的关系）
================================
设一段语料共 :math:`T` 个 token，模型在第 :math:`t` 个位置对词表
:math:`\\mathcal V` 输出 logits :math:`z_t\\in\\mathbb R^{|\\mathcal V|}`，
真实 token 为 :math:`y_t`。

1. **Log-softmax**（数值稳定写法，避免先 exp 再 log 的上溢）：

   .. math::
       \\log p_t(v) = z_t(v) - \\log\\sum_{v'}\\exp z_t(v')
       = z_t(v) - \\mathrm{logsumexp}(z_t)

2. **交叉熵 / NLL（负对数似然）**——把所有位置取平均：

   .. math::
       \\mathrm{NLL} = \\mathrm{CE} =
       -\\frac{1}{T}\\sum_{t=1}^{T}\\log p_t(y_t)

3. **困惑度**——把 NLL 取指数，量纲回到"词表大小"：

   .. math::
       \\mathrm{PPL} = \\exp(\\mathrm{NLL})
       = \\exp\\!\\left(-\\frac{1}{T}\\sum_t\\log p_t(y_t)\\right)
       = \\prod_{t=1}^{T} p_t(y_t)^{-1/T}

几个必须记住的结论
------------------
- 当模型对每个位置的真实 token 都给概率 1 时，NLL=0，PPL=1（完美）。
- 当模型在大小为 :math:`V` 的词表上完全随机（均匀分布）时，
  每个 token 概率都是 :math:`1/V`，NLL=log V，**PPL=V**。
  这就是为什么"PPL 不可能低于 1"、"随机基线 PPL 约等于词表大小"。
- 跨文档聚合时，**必须按 token 数加权**（直接把所有位置 log p 拼起来求平均），
  而不是把每篇文档各自的 PPL 取算术平均——后者会短文档权重过高。

本模块用 numpy 实现，输入接受 ``np.ndarray``；PyTorch 调用方请先
``logits.detach().cpu().numpy()``，张量形状注释里给出对应形状。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["log_softmax", "cross_entropy", "nll", "perplexity", "corpus_ppl"]


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 log-softmax。

    Args:
        logits: 形状 ``[..., V]`` 的浮点数组（本项目里通常是 ``[T, V]``）。
        axis: 词表维度。

    Returns:
        与 ``logits`` 同形状的 ``log p``，在 ``axis`` 维上和为 0（取 exp 后和为 1）。
    """
    logits = np.asarray(logits, dtype=np.float64)
    # 减最大值：不改变 softmax 结果，但避免 exp 上溢
    m = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - m
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + m
    return logits - lse


def cross_entropy(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """逐 token 交叉熵（即平均 NLL）。

    形状约定：

    - ``logits``: ``[T, V]``，第 t 行是模型在预测第 t 个 token 时的 logits。
    - ``targets``: ``[T]``，每个元素是 ``[0, V)`` 的整数索引。
    - ``mask``: 可选 ``[T]``，0 表示该位置不计入（比如 padding 被移位后的 -100
      在调用方这里已提前过滤成 mask=0）。

    Returns:
        标量 CE。空序列抛 ``ValueError``。
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是 [T, V]，收到 shape={logits.shape}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets 必须是 [T] 且与 logits 行数一致，收到 {targets.shape}"
        )

    logp = log_softmax(logits, axis=-1)  # [T, V]
    # 高级索引取出每个位置真实 token 的 log p
    token_logp = logp[np.arange(logits.shape[0]), targets]  # [T]

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != token_logp.shape:
            raise ValueError("mask 形状与 token 数不一致")
        denom = float(mask.sum())
        if denom == 0.0:
            raise ValueError("mask 全为 0，没有可计分的 token")
        return float(-(token_logp * mask).sum() / denom)

    if token_logp.size == 0:
        raise ValueError("没有可计分的 token")
    return float(-token_logp.mean())


def nll(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    """NLL 是交叉熵的别名——语言模型文档里这两个词几乎可互换。"""
    return cross_entropy(logits, targets, mask=mask)


def perplexity(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """PPL = exp(CE)。"""
    ce = cross_entropy(logits, targets, mask=mask)
    return float(math.exp(ce))


def corpus_ppl(
    examples: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, int, float]:
    """跨文档（或跨 batch）按 token 数加权的语料级 PPL。

    正确做法：把所有文档的 log p 全部拼起来，除以总 token 数，再取 exp。
    **不要**先算每篇文档的 PPL 再平均。

    Args:
        examples: 每个元素是 ``(logits[T, V], targets[T])``。

    Returns:
        ``(ppl, total_tokens, mean_nll)``。
    """
    total_nll = 0.0
    total_tokens = 0
    for logits, targets in examples:
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        logp = log_softmax(logits, axis=-1)
        token_logp = logp[np.arange(logits.shape[0]), targets]
        total_nll += float(-token_logp.sum())
        total_tokens += int(token_logp.shape[0])

    if total_tokens == 0:
        raise ValueError("corpus_ppl 收到的总 token 数为 0")
    mean_nll = total_nll / total_tokens
    return float(math.exp(mean_nll)), total_tokens, float(mean_nll)

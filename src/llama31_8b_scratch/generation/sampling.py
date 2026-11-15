"""解码采样策略手写实现。

覆盖的策略
----------
- **Temperature（温度）**：对 logits 做缩放

  .. math::

      z_i' = \\frac{z_i}{T}

  温度 ``T < 1`` 放大分布尖锐度（更确定），``T > 1`` 拉平分布（更多样）；
  ``T = 0`` 退化为贪心解码（argmax）。

- **Top-k**：只保留概率最高的 ``k`` 个候选，其余 logits 置 ``-inf``。
- **Top-p（核采样）**：按概率降序累计，只保留累计概率不超过 ``p`` 的最小候选集合。
- **重复惩罚（repetition penalty）**：对已出现过的 token 调整 logits，
  降低重复生成概率：

  .. math::

      z_i' = \\begin{cases}
          z_i / p, & z_i \\ge 0 \\\\
          z_i \\cdot p, & z_i < 0
      \\end{cases}

  其中 ``p`` 为惩罚系数（``p > 1`` 抑制重复）。

形状约定
--------
- 输入 ``logits``: ``[B, V]``（最后一步的词表 logits）；
- ``input_ids``: ``[B, T]`` 已生成序列，用于重复惩罚；
- 输出 token 下标: ``[B, 1]``。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """按温度缩放 logits；温度非正视为贪心分支，交由上层处理。"""
    if temperature is None or temperature <= 0:
        return logits
    return logits / temperature


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k 过滤：保留概率最高的 k 个 token，其余置 ``-inf``。

    Args:
        logits: ``[B, V]``。
        top_k: 保留候选数；``<= 0`` 表示不启用。
    """
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.shape[-1])
    # 第 k 大的分数作为阈值，低于阈值的候选全部屏蔽
    kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    mask = logits < kth_value
    return logits.masked_fill(mask, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p（核采样）过滤：保留累计概率不超过 ``p`` 的候选集合。

    Args:
        logits: ``[B, V]``。
        top_p: 累计概率阈值；``>= 1.0`` 表示不启用。
    """
    if top_p is None or top_p >= 1.0:
        return logits

    # 按 logits 降序排序，并计算 softmax 概率的累计和
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # 第一个使累计概率超过 p 的位置及其后全部屏蔽；
    # 注意保留“刚好跨过 p”的那个 token，因此把 mask 向后平移一位。
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

    # 还原回原始词表顺序
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    penalty: float,
) -> torch.Tensor:
    """重复惩罚：对历史出现过的 token 按 HF 口径调整 logits。

    Args:
        logits: ``[B, V]``。
        input_ids: ``[B, T]`` 历史 token；可空。
        penalty: 惩罚系数，``1.0`` 表示不惩罚。
    """
    if penalty is None or penalty == 1.0 or input_ids is None:
        return logits

    score = logits.clone()
    for batch_idx in range(score.shape[0]):
        past_tokens = input_ids[batch_idx]
        past_tokens = past_tokens[past_tokens >= 0]  # 忽略 padding/屏蔽下标
        if past_tokens.numel() == 0:
            continue
        past_logits = score[batch_idx, past_tokens]
        # 正 logits 除以惩罚（更负向），负 logits 乘以惩罚（更负向）
        adjusted = torch.where(
            past_logits < 0,
            past_logits * penalty,
            past_logits / penalty,
        )
        score[batch_idx, past_tokens] = adjusted
    return score


def sample_next_token(
    logits: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """根据一组 logits 采样下一个 token。

    Args:
        logits: ``[B, V]``。
        input_ids: 历史序列，供重复惩罚使用。
        temperature: 温度；``0`` 或 ``greedy=True`` 时取 argmax。
        top_k: top-k 候选数。
        top_p: 核采样累计概率阈值。
        repetition_penalty: 重复惩罚系数。
        greedy: 强制贪心解码。

    Returns:
        ``[B, 1]`` token 下标张量。
    """
    logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

    if greedy or temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

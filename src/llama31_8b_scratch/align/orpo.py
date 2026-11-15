"""ORPO（Odds Ratio Preference Optimization）手写实现。

论文：Hong & Lee, 2024, *ORPO: Monolithic Preference Optimization without
Reference Model*。

ORPO 的关键差异：**单模型、无需参考模型**，在同一次反向传播里同时做
SFT 与偏好对齐。损失由两部分组成：

1. **SFT 损失**：对 chosen 回答做教师强制的负对数似然：

   .. math::

       \\mathcal{L}_{SFT}
       = -\\frac{1}{|\\mathcal{M}_w|}\\sum_{(b,t)\\in\\mathcal{M}_w}
       \\log \\pi_\\theta(y_t^w \\mid y_{<t}^w, x)

2. **赔率比（Odds Ratio, OR）损失**：先把序列平均对数概率转成概率
   :math:`p = \\exp(\\bar{\\ell})`，再取对数赔率

   .. math::

       \\log O(y|x) = \\log \\frac{p}{1-p}
       = \\bar{\\ell} - \\log(1 - \\exp(\\bar{\\ell}))

   成对 OR 损失：

   .. math::

       \\mathcal{L}_{OR}
       = -\\frac{1}{B}\\sum_i \\log \\sigma\\Big(
       \\beta\\big[\\log O(y_i^w|x_i) - \\log O(y_i^l|x_i)\\big]
       \\Big)

总损失：

.. math::

    \\mathcal{L}_{ORPO} = \\mathcal{L}_{SFT} + \\lambda_{OR}\\,\\mathcal{L}_{OR}
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from llama31_8b_scratch.align.utils import masked_mean


def sequence_mean_logprobs(
    token_logps: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """把逐位置对数概率平均为每条序列的“平均 token 对数概率”。

    .. math::

        \\bar{\\ell} = \\frac{1}{|\\mathcal{M}|}\\sum_{t} \\ell_t m_t

    ORPO 用平均（而非求和）对数概率转赔率，避免序列长度直接进入 odds 尺度。
    """
    return masked_mean(token_logps, mask, dim=-1)


def log_odds(log_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """由平均对数概率计算对数赔率。

    设 :math:`p = \\exp(\\bar{\\ell})`，则

    .. math::

        \\log O = \\log \\frac{p}{1-p}
        = \\bar{\\ell} - \\log(1 - p)
        = \\bar{\\ell} - \\log1p(-p)

    Args:
        log_probs: 序列平均对数概率 ``[B]``，取值应小于 0。
        eps: 把 ``p`` 上界钳制到 ``1-eps``，防止 ``log1p`` 接收非正数。
    """
    p = torch.exp(log_probs).clamp(max=1.0 - eps)
    return log_probs - torch.log1p(-p)


def orpo_loss(
    chosen_mean_logps: torch.Tensor,
    rejected_mean_logps: torch.Tensor,
    chosen_nll: torch.Tensor,
    beta: float = 0.1,
    lambda_or: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """ORPO 组合损失。

    Args:
        chosen_mean_logps: chosen 序列平均 token 对数概率 ``[B]``。
        rejected_mean_logps: rejected 序列平均 token 对数概率 ``[B]``。
        chosen_nll: chosen 的 SFT 负对数似然标量（已按有效 token 平均）。
        beta: 赔率比温度 :math:`\\beta`。
        lambda_or: OR 损失权重 :math:`\\lambda_{OR}`。

    Returns:
        字典，包含 ``loss``、``sft_loss``、``or_loss``、
        ``log_odds_chosen``、``log_odds_rejected``、``accuracy``。
    """
    if chosen_mean_logps.shape != rejected_mean_logps.shape:
        raise ValueError("chosen/rejected 平均对数概率形状必须一致")

    # 对数赔率：log O_w, log O_l
    log_odds_chosen = log_odds(chosen_mean_logps)
    log_odds_rejected = log_odds(rejected_mean_logps)

    # β · (log O_w - log O_l)
    odds_ratio_logits = beta * (log_odds_chosen - log_odds_rejected)
    or_loss = -F.logsigmoid(odds_ratio_logits).mean()

    total = chosen_nll + lambda_or * or_loss

    with torch.no_grad():
        accuracy = (odds_ratio_logits > 0).to(torch.float32).mean()

    return {
        "loss": total,
        "sft_loss": chosen_nll,
        "or_loss": or_loss,
        "log_odds_chosen": log_odds_chosen,
        "log_odds_rejected": log_odds_rejected,
        "odds_ratio_logits": odds_ratio_logits,
        "accuracy": accuracy,
    }

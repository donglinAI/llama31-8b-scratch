"""奖励模型（Reward Model, RM）手写实现。

奖励模型在 SFT 模型 backbone 顶部接一个标量打分头，对给定 prompt-response
输出一个标量分数 :math:`r(x,y)`。训练数据为偏好对
``{prompt, chosen, rejected}``（见 SPEC 5.3），目标是让 chosen 的分数高于
rejected。

核心损失为 Bradley-Terry 成对损失（pairwise loss）：

.. math::

    \\mathcal{L}_{RM}
    = -\\frac{1}{N}\\sum_{i=1}^{N}
    \\log \\sigma\\big(r(x_i, y_i^w) - r(x_i, y_i^l)\\big)

其中 :math:`\\sigma` 为 sigmoid，:math:`y^w` 为 chosen（更好回答），
:math:`y^l` 为 rejected（更差回答）。

数值稳定写法：直接使用 ``F.logsigmoid``，避免先算 sigmoid 再取对数导致的
大数饱和与下溢；当差值很大时 ``logsigmoid`` 仍保持良好数值范围。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry 成对损失。

    Args:
        chosen_scores: 形状 ``[B]``，chosen 回答的 RM 打分 :math:`r_w`。
        rejected_scores: 形状 ``[B]``，rejected 回答的 RM 打分 :math:`r_l`。
        margin: 可选间隔 :math:`m`；要求 :math:`r_w - r_l \\ge m`，
            差值项变为 :math:`r_w - r_l - m`。默认 0。

    Returns:
        标量损失：

        .. math::

            \\mathcal{L} = -\\frac{1}{B}\\sum_i
            \\log \\sigma(r_{w,i} - r_{l,i} - m)
    """
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError(
            f"chosen/rejected 打分形状不一致: {tuple(chosen_scores.shape)} vs "
            f"{tuple(rejected_scores.shape)}"
        )
    # 差值 d = r_w - r_l - m；logsigmoid(d) 在 d 为大负数时也不会下溢到 -inf 之外
    diff = chosen_scores - rejected_scores - margin
    loss = -F.logsigmoid(diff).mean()
    return loss


def pairwise_accuracy(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """成对排序准确率：chosen 分数严格高于 rejected（含 margin）记为正确。

    Returns:
        标量 ``[0, 1]`` 准确率。
    """
    correct = (chosen_scores - rejected_scores - margin) > 0
    return correct.to(torch.float32).mean()


def compute_pairwise_metrics(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """一次性返回 RM 训练常用指标（损失、准确率、分数均值）。"""
    return {
        "loss": pairwise_loss(chosen_scores, rejected_scores, margin=margin),
        "accuracy": pairwise_accuracy(chosen_scores, rejected_scores, margin=margin),
        "chosen_mean": chosen_scores.mean(),
        "rejected_mean": rejected_scores.mean(),
        "score_gap": (chosen_scores - rejected_scores).mean(),
    }

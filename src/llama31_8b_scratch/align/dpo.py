"""DPO（Direct Preference Optimization）手写实现。

论文：Rafailov et al., 2023, *Direct Preference Optimization: Your Language
Model is Secretly a Reward Model*。

DPO 跳过显式训练奖励模型与 RL 采样，直接在偏好对上优化策略。其核心是把
KL 约束的奖励最大化问题解析求解，得到策略相对参考模型的隐式奖励：

.. math::

    r(x, y) = \\beta\\Big[
    \\log \\pi_\\theta(y \\mid x) - \\log \\pi_{\\mathrm{ref}}(y \\mid x)
    \\Big]

成对损失（Bradley-Terry 形式）：

.. math::

    \\mathcal{L}_{DPO}
    = -\\mathbb{E}_{(x, y^w, y^l)}
    \\log \\sigma\\Big(
    r(x, y^w) - r(x, y^l)
    \\Big)

展开为对数概率形式：

.. math::

    \\mathcal{L}_{DPO}
    = -\\frac{1}{B}\\sum_i \\log \\sigma\\Big(
    \\beta\\big[
    (\\logp^\\theta_w - \\logp^{ref}_w)
    - (\\logp^\\theta_l - \\logp^{ref}_l)
    \\big]
    \\Big)

其中 :math:`\\beta` 控制策略偏离参考模型的强度，越大约束越紧。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """计算 DPO 成对损失。

    Args:
        policy_chosen_logps: 当前策略 chosen 序列对数概率 ``[B]``。
        policy_rejected_logps: 当前策略 rejected 序列对数概率 ``[B]``。
        ref_chosen_logps: 参考模型 chosen 序列对数概率 ``[B]``。
        ref_rejected_logps: 参考模型 rejected 序列对数概率 ``[B]``。
        beta: DPO 温度系数 :math:`\\beta`。
        label_smoothing: 标签平滑 :math:`\\epsilon \\in [0,1)`；损失变为
            :math:`-[(1-\\epsilon)\\log\\sigma(z) + \\epsilon\\log\\sigma(-z)]`，
            防止模型把偏好 margin 推到无穷大。

    Returns:
        字典，包含：

        - ``loss``：标量 DPO 损失；
        - ``logits``：未乘 β 之前的偏好差（即隐式奖励差 / β）；
        - ``rewards_chosen`` / ``rewards_rejected``：隐式奖励；
        - ``rewards_margin``：两者之差（= :math:`\\beta \\cdot logits`）；
        - ``chosen_rewards`` 等监控指标。
    """
    tensors = [
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ]
    if any(t.shape != policy_chosen_logps.shape for t in tensors[1:]):
        raise ValueError("四个序列对数概率张量形状必须一致")

    # 策略相对参考模型的对数概率差（chosen / rejected 各一条）
    policy_chosen_delta = policy_chosen_logps - ref_chosen_logps
    policy_rejected_delta = policy_rejected_logps - ref_rejected_logps

    # 隐式奖励 r(x,y) = β · (logπ_θ - logπ_ref)
    rewards_chosen = beta * policy_chosen_delta
    rewards_rejected = beta * policy_rejected_delta

    # DPO logit：β[(Δ_w) - (Δ_l)]
    dpo_logits = rewards_chosen - rewards_rejected

    if label_smoothing > 0.0:
        # 平滑目标：以 ε 概率把标签视为“反例”，对应 -log σ(-z)
        eps = label_smoothing
        loss = -(
            (1.0 - eps) * F.logsigmoid(dpo_logits)
            + eps * F.logsigmoid(-dpo_logits)
        ).mean()
    else:
        loss = -F.logsigmoid(dpo_logits).mean()

    with torch.no_grad():
        margin = rewards_chosen - rewards_rejected
        # 排序准确率：chosen 隐式奖励高于 rejected
        accuracy = (margin > 0).to(torch.float32).mean()

    return {
        "loss": loss,
        "logits": dpo_logits / beta,
        "rewards_chosen": rewards_chosen,
        "rewards_rejected": rewards_rejected,
        "rewards_margin": margin,
        "accuracy": accuracy,
    }


def implicit_reward(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """计算单条序列的 DPO 隐式奖励 :math:`r = \\beta(\\logp_\\theta - \\logp_{ref})`。"""
    return beta * (policy_logps - ref_logps)

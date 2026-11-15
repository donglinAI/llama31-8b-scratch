"""GRPO（Group Relative Policy Optimization）手写实现。

论文：DeepSeekMath, 2024, *DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open-Source Language Models*；DeepSeek-R1, 2025,
*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
Learning*。

GRPO 相对经典 PPO 的两点核心简化：

1. **去掉价值网络**：对同一个 prompt 采样一组（G 条）回答，用组内奖励的
   相对高低作为优势，无需 critic 估计基线；
2. **组内相对优势**：

   .. math::

       A_{i} = \\frac{r_i - \\mathrm{mean}_{j=1..G} r_j}
       {\\mathrm{std}_{j=1..G} r_j + \\varepsilon}

策略目标沿用 PPO 裁剪代理形式，但只在 token / 组维度平均，没有价值损失：

.. math::

    \\mathcal{L}_{GRPO}
    = -\\frac{1}{G}\\sum_{i=1}^{G}
    \\frac{1}{|\\mathcal{M}_i|}\\sum_{t\\in\\mathcal{M}_i}
    \\min\\big(
    \\rho_{i,t} A_i,\\;
    \\mathrm{clip}(\\rho_{i,t}, 1-\\varepsilon, 1+\\varepsilon) A_i
    \\big)
    + \\beta_{KL}\\,\\mathbb{D}_{KL}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}})

其中 :math:`\\rho_{i,t} = \\exp(\\logp^\\theta_{i,t} - \\logp^{old}_{i,t})`。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from llama31_8b_scratch.align.utils import compute_kl_penalty, masked_mean


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """组内相对优势标准化。

    Args:
        rewards: 形状 ``[P*G]``，先按 prompt 分组、组内连续排列的奖励。
        group_size: 每组回答数 :math:`G`。
        eps: 标准差除零保护。

    Returns:
        形状 ``[P*G]``，每个 prompt 组内零均值、单位方差的优势。
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"奖励数量 {rewards.numel()} 不能被组大小 {group_size} 整除"
        )
    num_prompts = rewards.numel() // group_size
    grouped = rewards.view(num_prompts, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - group_mean) / (group_std + eps)
    return advantages.view(-1)


def grpo_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    ref_logps: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.0,
    kl_estimator: str = "naive",
) -> Dict[str, torch.Tensor]:
    """GRPO 策略损失（无价值网络）。

    Args:
        new_logps: 当前策略逐位置对数概率 ``[N, T]``（N = P*G）。
        old_logps: 采样时旧策略对数概率 ``[N, T]``。
        advantages: 组内相对优势 ``[N]``，会沿时间维广播到 ``[N, T]``。
        mask: 有效位置掩码 ``[N, T]``。
        ref_logps: 参考模型对数概率 ``[N, T]``；``None`` 不计 KL。
        clip_ratio: 策略裁剪阈值 :math:`\\varepsilon`。
        kl_coef: KL 惩罚系数 :math:`\\beta_{KL}`。
        kl_estimator: KL 估计量 ``"naive"`` / ``"k3"``。

    Returns:
        字典，包含 ``loss``、``policy_loss``、``kl``、``ratio``。
    """
    if advantages.dim() == 1:
        # 优势是每条序列一个标量，广播到 token 维
        adv = advantages.unsqueeze(-1)
    else:
        adv = advantages

    # ρ = exp(new - old)
    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)

    surr1 = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surr2 = clipped_ratio * adv
    surrogate = -torch.min(surr1, surr2)
    policy_loss = masked_mean(surrogate, mask)

    kl_value = torch.zeros((), device=new_logps.device, dtype=new_logps.dtype)
    kl_penalty = torch.zeros_like(kl_value)
    if ref_logps is not None and kl_coef > 0.0:
        kl_penalty, kl_value = compute_kl_penalty(
            new_logps, ref_logps, mask, kl_coef=kl_coef, estimator=kl_estimator
        )

    loss = policy_loss + kl_penalty

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "kl": kl_value,
        "kl_penalty": kl_penalty,
        "ratio": ratio,
    }

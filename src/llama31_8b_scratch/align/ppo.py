"""经典 RLHF 的 PPO 手写实现（Schulman et al., 2017）。

本模块覆盖 PPO 在 RLHF 中的全部教学要素：

1. 价值头（value head）：在 backbone 隐状态上接标量回归头估计状态价值；
2. 广义优势估计（GAE, Schulman et al., 2016）：从 TD 残差递推得到优势与回报；
3. 裁剪策略目标（clipped surrogate objective）；
4. 价值函数裁剪（value clip）；
5. 相对参考模型的 KL 惩罚；
6. 奖励 / 优势白化。

所有公式均给出无跳步推导，张量形状在注释中标明。工业规模化的 TRL
``PPOTrainer`` 封装见 :mod:`llama31_8b_scratch.align.trl_wrappers`。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from llama31_8b_scratch.align.utils import (
    compute_kl_penalty,
    masked_mean,
    normalize_advantages,
)


class ValueHead(nn.Module):
    """价值头：把 backbone 隐状态映射为标量状态价值 :math:`V(s)`。

    结构为单层线性投影（无偏置可选），输出形状 ``[B, T]``。
    """

    def __init__(self, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            hidden_states: 形状 ``[B, T, H]``。

        Returns:
            形状 ``[B, T]`` 的价值估计。
        """
        return self.head(hidden_states).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计（Generalized Advantage Estimation）。

    **无跳步推导**

    设轨迹为 :math:`r_0, r_1, \\dots, r_{T-1}`，价值估计
    :math:`V_0, \\dots, V_{T-1}`，终止标记 :math:`d_t \\in \\{0,1\\}`
    （``d_t=1`` 表示 t 步之后回合结束）。

    第一步，时序差分残差（TD residual）：

    .. math::

        \\delta_t = r_t + \\gamma\\, V_{t+1}(1-d_t) - V_t

    其中回合终止时不引导下一状态价值，故乘 :math:`(1-d_t)`；轨迹末尾
    :math:`V_T = 0`（不做外部 bootstrap）。

    第二步，GAE 用指数加权把多步 TD 残差合并为优势：

    .. math::

        A_t = \\delta_t + \\gamma\\lambda (1-d_t) A_{t+1},
        \\quad A_T = 0

    展开可见 :math:`A_t` 是 :math:`\\delta_t, \\delta_{t+1}, \\dots` 的
    :math:`(\\gamma\\lambda)` 指数加权和，偏差-方差由 :math:`\\lambda` 折中。

    第三步，回报（return）由优势加回价值得到：

    .. math::

        R_t = A_t + V_t

    Args:
        rewards: 形状 ``[B, T]``，每步奖励。
        values: 形状 ``[B, T]``，价值头对每步状态的估计。
        dones: 形状 ``[B, T]``，终止标记（1 表示回合结束）。
        gamma: 折扣因子 :math:`\\gamma`。
        lam: GAE 衰减 :math:`\\lambda`。

    Returns:
        ``(advantages, returns)``，形状均为 ``[B, T]``。
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"rewards/values/dones 形状应一致: {tuple(rewards.shape)}, "
            f"{tuple(values.shape)}, {tuple(dones.shape)}"
        )

    batch, seq_len = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    advantages = torch.zeros(batch, seq_len, device=device, dtype=dtype)
    last_gae = torch.zeros(batch, device=device, dtype=dtype)

    for t in reversed(range(seq_len)):
        # 下一时刻价值：轨迹末尾 V_T 取 0
        if t + 1 < seq_len:
            next_value = values[:, t + 1]
        else:
            next_value = torch.zeros(batch, device=device, dtype=dtype)

        # 非终止掩码：d_t=1 时不再向后续引导
        non_terminal = 1.0 - dones[:, t].to(dtype)

        # δ_t = r_t + γ V_{t+1} (1-d_t) - V_t
        delta = rewards[:, t] + gamma * next_value * non_terminal - values[:, t]

        # A_t = δ_t + γλ (1-d_t) A_{t+1}
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[:, t] = last_gae

    # R_t = A_t + V_t
    returns = advantages + values
    return advantages, returns


def clipped_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PPO 裁剪策略目标（clipped surrogate objective）。

    重要性比：

    .. math::

        \\rho_t = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\mathrm{old}}(a_t|s_t)}
        = \\exp\\big(\\logp_\\theta - \\logp_{\\mathrm{old}}\\big)

    裁剪代理目标（最大化）：

    .. math::

        L^{CLIP} = \\mathbb{E}_t\\big[
        \\min\\big(\\rho_t A_t,\\;
        \\mathrm{clip}(\\rho_t, 1-\\varepsilon, 1+\\varepsilon) A_t\\big)\\big]

    代码中取负号作为最小化损失。当优势 :math:`A_t>0` 时，提升概率被截断在
    :math:`1+\\varepsilon`；当 :math:`A_t<0` 时，压低概率被截断在
    :math:`1-\\varepsilon`，防止单步更新幅度过大。

    Args:
        new_logps: 当前策略逐位置对数概率 ``[B, T]``。
        old_logps: 采样时旧策略逐位置对数概率 ``[B, T]``。
        advantages: GAE 优势 ``[B, T]``。
        mask: 有效位置掩码 ``[B, T]``。
        clip_ratio: 裁剪阈值 :math:`\\varepsilon`。

    Returns:
        ``(policy_loss, ratio)``：标量策略损失与重要性比张量。
    """
    # ρ = exp(new - old)；log 域相减避免直接算概率比值下溢
    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surr2 = clipped_ratio * advantages

    # min 之后取负：最小化 -L^CLIP
    surrogate = -torch.min(surr1, surr2)
    policy_loss = masked_mean(surrogate, mask)
    return policy_loss, ratio


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """价值函数裁剪损失。

    先把新价值相对旧价值的偏差裁剪到 :math:`[-\\varepsilon_v, \\varepsilon_v]`，
    再构造裁剪后的价值预测：

    .. math::

        V^{clipped} = V_{old} + \\mathrm{clip}(V_\\theta - V_{old},
        -\\varepsilon_v, \\varepsilon_v)

    价值损失取两种平方误差的较大者：

    .. math::

        L^{VF} = \\mathbb{E}_t\\big[\\max\\big(
        (V_\\theta - R_t)^2,\\;
        (V^{clipped} - R_t)^2\\big)\\big]

    裁剪价值头是为了避免价值网络在一轮内剧烈漂移，进而破坏策略-价值的平衡。
    """
    values_clipped = old_values + torch.clamp(
        values - old_values, -clip_ratio, clip_ratio
    )
    vf_loss1 = (values - returns).pow(2)
    vf_loss2 = (values_clipped - returns).pow(2)
    vf_loss = torch.max(vf_loss1, vf_loss2)
    return masked_mean(vf_loss, mask)


def ppo_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    ref_logps: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    value_clip_ratio: float = 0.2,
    value_coef: float = 1.0,
    kl_coef: float = 0.05,
    kl_estimator: str = "naive",
    whiten_advantages: bool = True,
) -> Dict[str, torch.Tensor]:
    """PPO 一轮更新的组合损失。

    总损失（最小化）：

    .. math::

        L = L^{CLIP} + c_v L^{VF} + \\beta_{KL} \\,
        \\mathbb{D}_{KL}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}})

    Args:
        new_logps: 当前策略逐位置对数概率 ``[B, T]``。
        old_logps: 采样时旧策略对数概率 ``[B, T]``。
        advantages: GAE 优势 ``[B, T]``。
        values: 当前价值头预测 ``[B, T]``。
        old_values: 采样时旧价值头预测 ``[B, T]``。
        returns: GAE 回报 ``[B, T]``。
        mask: 有效位置掩码 ``[B, T]``。
        ref_logps: 参考模型对数概率 ``[B, T]``；``None`` 时不计 KL。
        clip_ratio: 策略裁剪阈值。
        value_clip_ratio: 价值裁剪阈值。
        value_coef: 价值损失权重 :math:`c_v`。
        kl_coef: KL 惩罚系数 :math:`\\beta_{KL}`。
        kl_estimator: KL 估计量，``"naive"`` / ``"k3"``。
        whiten_advantages: 是否在损失前对优势做批内白化。

    Returns:
        包含 ``loss``、``policy_loss``、``value_loss``、``kl``、``ratio`` 的字典。
    """
    adv = advantages
    if whiten_advantages:
        adv = normalize_advantages(adv, mask=mask)

    policy_loss, ratio = clipped_policy_loss(
        new_logps, old_logps, adv, mask, clip_ratio=clip_ratio
    )
    value_loss = clipped_value_loss(
        values, old_values, returns, mask, clip_ratio=value_clip_ratio
    )

    kl_value = torch.zeros((), device=new_logps.device, dtype=new_logps.dtype)
    kl_penalty = torch.zeros_like(kl_value)
    if ref_logps is not None and kl_coef > 0.0:
        kl_penalty, kl_value = compute_kl_penalty(
            new_logps, ref_logps, mask, kl_coef=kl_coef, estimator=kl_estimator
        )

    loss = policy_loss + value_coef * value_loss + kl_penalty

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "kl": kl_value,
        "kl_penalty": kl_penalty,
        "ratio": ratio,
    }

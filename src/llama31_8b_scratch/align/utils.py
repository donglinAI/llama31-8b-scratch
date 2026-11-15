"""对齐算法通用数学工具。

本模块只依赖 ``torch`` 与 ``numpy``，提供 RM / PPO / DPO / ORPO / GRPO 共用的：

- 序列对数概率提取（``gather_logprobs``）；
- 掩码平均 / 掩码求和；
- 优势（advantage）归一化；
- 奖励白化（whitening）；
- 策略相对参考模型的 KL 散度估计量。

所有公式在注释中给出 KaTeX 形式，并与代码符号逐一对齐。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

#: 标签中表示“忽略位置”的默认值（与 PyTorch ``cross_entropy`` 的 ``ignore_index`` 对齐）
DEFAULT_IGNORE_INDEX: int = -100


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从词表 logits 中按标签抽取每个位置的对数概率。

    数学定义（逐位置）：

    .. math::

        \\ell_{b,t} = \\log \\pi_\\theta(y_t \\mid y_{<t}, x)
        = \\log \\frac{\\exp(z_{b,t,y_t})}{\\sum_v \\exp(z_{b,t,v})}

    其中 :math:`z` 为 ``logits``。忽略位置（``labels == ignore_index``）的输出概率置 0，
    避免其进入后续求和 / 平均。

    Args:
        logits: 形状 ``[B, T, V]``，未经过 softmax 的词表打分。
        labels: 形状 ``[B, T]``，目标 token id；忽略位置填 ``ignore_index``。
        ignore_index: 被掩码掉的标签值。

    Returns:
        ``(token_logps, mask)``：

        - ``token_logps``：形状 ``[B, T]``，逐位置对数概率，忽略位置为 0；
        - ``mask``：形状 ``[B, T]`` 的布尔张量，``True`` 表示有效位置。
    """
    if logits.dim() != 3:
        raise ValueError(f"logits 形状应为 [B, T, V]，实际为 {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels 形状 {tuple(labels.shape)} 与 logits 前两维 "
            f"{tuple(logits.shape[:2])} 不一致"
        )

    # 先在词表维做 log-softmax，得到逐 token 的对数概率分布
    log_probs = F.log_softmax(logits, dim=-1)

    # 有效位置掩码；忽略位置先填 0，防止 gather 到非法 id 时报错
    mask = labels != ignore_index
    labels_safe = labels.masked_fill(~mask, 0)

    # gather 需要最后一维形状为 [B, T, 1] 的索引，抽取后压回最后一维
    gathered = log_probs.gather(
        dim=-1, index=labels_safe.unsqueeze(-1)
    ).squeeze(-1)

    # 忽略位置的对数概率清零，保证不污染序列求和
    token_logps = gathered * mask
    return token_logps, mask


def sequence_logprobs(token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把逐位置对数概率沿时间维求和，得到每条序列的序列对数概率。

    数学定义：

    .. math::

        \\log \\pi_\\theta(y \\mid x) = \\sum_{t=1}^{T} \\ell_{b,t} \\, m_{b,t}

    Args:
        token_logps: 形状 ``[B, T]``，由 :func:`gather_logprobs` 得到。
        mask: 形状 ``[B, T]``，有效位置掩码。

    Returns:
        形状 ``[B]``，每条序列的对数概率之和。
    """
    return (token_logps * mask.to(token_logps.dtype)).sum(dim=-1)


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按掩码对 ``values`` 求和，掩码位置不计入。"""
    return (values * mask.to(values.dtype)).sum()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """按掩码求平均，避免把 padding / 忽略位置计入分母。

    当 ``dim`` 为 ``None`` 时对全部有效位置求标量平均；否则沿指定维做掩码平均。
    """
    dtype = values.dtype
    mask_f = mask.to(dtype)
    summed = (values * mask_f).sum(dim=dim)
    count = mask_f.sum(dim=dim)
    return summed / (count + eps)


def normalize_advantages(
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """批内优势归一化：减均值、除标准差，使优势零均值、单位方差。

    数学定义：

    .. math::

        \\tilde{A}_t = \\frac{A_t - \\mu_A}{\\sigma_A + \\varepsilon}

    Args:
        advantages: 任意形状的优势张量。
        mask: 与 ``advantages`` 同形状的有效位置掩码；``None`` 表示全部有效。
        eps: 防止标准差除零的小常数。

    Returns:
        归一化后的优势张量，形状不变。
    """
    if mask is None:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        normalized = (advantages - mean) / (std + eps)
        return normalized

    mean = masked_mean(advantages, mask)
    std = masked_mean(advantages.pow(2), mask).sub(mean.pow(2)).clamp(min=0).sqrt()
    normalized = (advantages - mean) / (std + eps)
    # 掩码位置清零，避免 padding 优势进入后续策略损失
    return normalized * mask.to(normalized.dtype)


def whiten(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """奖励 / 返回值白化（whitening）：批内减均值除标准差。

    与 :func:`normalize_advantages` 语义一致，单独命名以对齐 PPO 文档中
    “reward whitening” 的术语；返回值不改变原始量纲方向，仅做标准化。
    """
    return normalize_advantages(values, mask=mask, eps=eps)


def kl_token_naive(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """前向 KL 的朴素逐位置估计：

    .. math::

        \\mathbb{D}_{KL}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}})
        \\approx \\frac{1}{|\\mathcal{M}|} \\sum_{(b,t)\\in\\mathcal{M}}
        \\big(\\log \\pi_\\theta(y_t|x) - \\log \\pi_{\\mathrm{ref}}(y_t|x)\\big)

    该估计是 Schulman 在 PPO 笔记中给出的 “k1” 估计量，计算最廉价，
    但在策略偏离参考模型较远时可能有偏。

    Args:
        logp_policy: 策略模型逐位置对数概率 ``[B, T]``。
        logp_ref: 参考模型逐位置对数概率 ``[B, T]``。
        mask: 有效位置掩码 ``[B, T]``。

    Returns:
        标量 KL 估计值。
    """
    per_token = logp_policy - logp_ref
    return masked_mean(per_token, mask)


def kl_token_k3(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """前向 KL 的 k3 估计量（Schulman, 2017），对策略偏移更稳健。

    令重要性比 :math:`\\rho = \\pi_\\theta / \\pi_{\\mathrm{ref}}`，则

    .. math::

        \\hat{\\mathbb{D}}_{KL}
        = \\frac{1}{|\\mathcal{M}|}\\sum_{(b,t)\\in\\mathcal{M}}
        \\big(\\rho - 1 - \\log \\rho\\big)

    其中 :math:`\\log \\rho = \\logp_\\theta - \\logp_{\\mathrm{ref}}`。
    该估计量在 :math:`\\rho=1` 附近二阶展开为 :math:`\\tfrac12(\\rho-1)^2`，
    恒非负，避免朴素估计量出现负值的问题。
    """
    log_ratio = logp_policy - logp_ref
    ratio = torch.exp(log_ratio)
    per_token = ratio - 1.0 - log_ratio
    return masked_mean(per_token, mask, eps=eps)


def compute_kl_penalty(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    mask: torch.Tensor,
    kl_coef: float,
    estimator: str = "naive",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算加权 KL 惩罚项。

    Args:
        logp_policy: 策略逐位置对数概率 ``[B, T]``。
        logp_ref: 参考模型逐位置对数概率 ``[B, T]``。
        mask: 有效位置掩码。
        kl_coef: KL 惩罚系数 :math:`\\beta_{KL}`。
        estimator: ``"naive"``（k1）或 ``"k3"``。

    Returns:
        ``(penalty, kl_value)``：加权后的惩罚项与未加权 KL 估计值。
    """
    if estimator == "naive":
        kl_value = kl_token_naive(logp_policy, logp_ref, mask)
    elif estimator == "k3":
        kl_value = kl_token_k3(logp_policy, logp_ref, mask)
    else:
        raise ValueError(f"未知 KL 估计量: {estimator!r}")
    penalty = kl_coef * kl_value
    return penalty, kl_value

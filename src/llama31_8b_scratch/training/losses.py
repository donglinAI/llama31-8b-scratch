"""因果语言模型的损失函数：交叉熵 + 可选 z-loss，讲清 label 移位与 -100 掩码。

一、为什么要 label 移位（label shift）？

因果 LM（Causal LM）的训练目标是：给定 ``x_1, ..., x_t``，预测下一个 token ``x_{t+1}``。
也就是说，位置 ``i`` 的隐状态应当预测位置 ``i+1`` 的 token。

模型输出 logits 的形状是 ``[B, T, V]``：
- ``logits[:, i, :]`` 是读完 ``x_1, ..., x_i`` 之后对下一个 token 的分布；
- 它应当去匹配 ``labels[:, i+1]``（而不是 ``labels[:, i]``）。

因此标准写法是：

.. code-block:: python

    F.cross_entropy(
        logits[..., :-1, :].reshape(-1, V),   # 位置 0..T-2 的预测
        labels[..., 1:].reshape(-1),          # 位置 1..T-1 的真值
    )

这一步在 SPEC 5.5 中被规定为"统一在模型/loss 内部完成"，数据集与 collator 只负责
返回等长的 ``input_ids`` / ``labels``，不做任何移位。

二、为什么用 -100 作为忽略标签？

PyTorch 的 ``F.cross_entropy`` 有一个约定俗成的 ``ignore_index=-100``：凡是 labels 中
等于 -100 的位置，其损失被置 0、且不参与分母求和。这个约定在 HF 生态里被广泛复用
（``transformers``、``trl``、``accelerate`` 都默认 -100）。

它的两个典型用途：
1. **padding**：一个 batch 里短序列右侧补的 pad token 不应产生梯度；
2. **SFT 只学回答**：``SFTCollator`` 会把 prompt 部分（system/user）的 label 全部置
   -100，只让 assistant 回答部分产生损失——否则模型会被强迫去拟合 prompt 的分布，
   那不是我们想要的。

三、z-loss 是什么、为什么加？

z-loss 来自 Google DeepMind / Gemini 团队的论文 "Scaling Laws for and of Compute"
与 "Cross-entropy z-loss" 实践：在大尺度训练时，softmax 分母
``Z(x) = logsumexp(logits)`` 有时会在某些层异常增大，导致训练后期梯度尖峰、loss
曲线出现 spike。z-loss 就是对 ``Z(x)`` 本身加一个 L2 惩罚：

.. math::
    L_{z} = \\beta \\cdot \\mathbb{E}\\!\\big[ \\log Z(x)^2 \\big]

其中 ``log Z = torch.logsumexp(logits, dim=-1)``。它只在 ``z_loss_coeff > 0`` 时启用，
对正常尺度模型影响极小，但在 70B 级以上、bf16 训练时能显著稳定 loss。我们默认系数
取 0（关闭），工业 8B 以上配置可开 1e-4 ~ 1e-3。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["causal_lm_loss", "z_loss"]

#: PyTorch / HF 生态约定的"忽略标签"，padding 与 prompt 掩码统一用这个值。
IGNORE_INDEX = -100


def z_loss(logits: torch.Tensor, z_loss_coeff: float = 0.0) -> torch.Tensor:
    """计算可选的 z-loss 惩罚项。

    Args:
        logits: 形状 ``[..., V]`` 的未归一化 logits（建议先做 label 移位再传入）。
        z_loss_coeff: 系数 ``beta``；为 0 时返回一个标量 0，省掉 logsumexp 计算。

    Returns:
        标量张量：``beta * mean(logsumexp(logits, dim=-1) ** 2)``。
    """
    if z_loss_coeff <= 0.0:
        # 返回与 logits 同设备同 dtype 的 0，保证计算图可连
        return logits.sum() * 0.0

    # logsumexp 在最后一维（词表维）上：得到每个预测位置的 log Z，形状 [..., 1] -> [...]
    log_z = torch.logsumexp(logits.float(), dim=-1)
    # 平方后对所有位置取均值，再乘 beta
    return z_loss_coeff * (log_z ** 2).mean().to(logits.dtype)


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    z_loss_coeff: float = 0.0,
) -> torch.Tensor:
    """因果 LM 的交叉熵损失，内部完成 label 移位与 -100 掩码。

    形状约定：
    - ``logits``: ``[B, T, V]``
    - ``labels``: ``[B, T]``，被 mask 的位置取值为 ``ignore_index``（默认 -100）。

    Args:
        logits: 模型输出的未归一化 logits。
        labels: 标签张量，padding / prompt 位置应为 ``ignore_index``。
        ignore_index: 被忽略的标签值，默认 -100。
        z_loss_coeff: z-loss 系数；0 表示关闭。

    Returns:
        标量损失张量，可直接 ``loss.backward()``。

    Note:
        这里**不**做 token 级平均后再乘序列长度——直接用 ``F.cross_entropy`` 的
        默认 ``mean``，它会自动只对非 -100 的位置求平均。这样在不同 batch 有效
        token 数不同时梯度尺度稳定，是 HF / Megatron / TRL 的统一约定。
    """
    # ---- 1) label 移位：位置 i 的 logits 预测位置 i+1 的 token ----
    # 把最后一维（词表 V）留给 cross_entropy；前两维 B、T 合并。
    # logits_shift: [B, T-1, V] -> [(B*(T-1)), V]
    # labels_shift: [B, T-1]      -> [(B*(T-1))]
    logits_shift = logits[..., :-1, :].contiguous()
    labels_shift = labels[..., 1:].contiguous()

    V = logits_shift.size(-1)
    # 展平：cross_entropy 接受 [N, V] 与 [N]
    loss_main = F.cross_entropy(
        logits_shift.view(-1, V),
        labels_shift.view(-1),
        ignore_index=ignore_index,
    )

    # ---- 2) 可选 z-loss（在移位后的 logits 上算，与主损失同位置） ----
    loss_z = z_loss(logits_shift, z_loss_coeff)

    return loss_main + loss_z

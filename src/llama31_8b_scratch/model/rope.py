r"""旋转位置编码（RoPE，Rotary Position Embedding）手写实现。

数学定义
--------
设单头维度 :math:`d = \\mathrm{head\\_dim}`，位置 :math:`t`，基频
:math:`\\theta_{\\text{base}}`。对第 :math:`i` 个成对维度（
:math:`i = 0, 1, \\dots, d/2 - 1`），定义角频率：

.. math::

    \\omega_i = \\theta_{\\text{base}}^{-\\frac{2i}{d}}

位置 :math:`t` 对应的相位为 :math:`t \\omega_i`。RoPE 把每个成对维度
:math:`(x_{2i}, x_{2i+1})` 当作二维向量，在其平面上旋转该相位：

.. math::

    \\begin{pmatrix}
        x'_{2i} \\\\
        x'_{2i+1}
    \\end{pmatrix}
    =
    \\begin{pmatrix}
        \\cos(t\\omega_i) & -\\sin(t\\omega_i) \\\\
        \\sin(t\\omega_i) &  \\cos(t\\omega_i)
    \\end{pmatrix}
    \\begin{pmatrix}
        x_{2i} \\\\ x_{2i+1}
    \\end{pmatrix}

等价的张量写法（HF / Llama 实现口径）是 ``rotate_half``：

.. math::

    \\mathrm{rotate\\_half}(x) = [ -x_{d/2:},\\ x_{:d/2} ]

从而：

.. math::

    x' = x \\odot \\cos(\\text{emb}) + \\mathrm{rotate\\_half}(x) \\odot \\sin(\\text{emb})

长上下文 scaling
----------------
- ``linear``：直接把频率除以缩放因子 :math:`s`，即 :math:`\\omega_i \\leftarrow \\omega_i / s`，
  相当于把位置轴整体拉长 :math:`s` 倍；
- ``dynamic``（NTK-by-parts）：提高 RoPE 基频，使高频维度（精确位置）不变、
  低频维度（粗粒度结构）被插值。等价实现：

  .. math::

      \\omega_i' = \\omega_i \\cdot s^{\\frac{2i}{d}}

- ``yarn``：完整 YaRN 包含修正维度区间、mscale 注意力缩放等，本模块在文档
  （``02_model/README.md``）中讲解完整数学，代码侧先回退到基础 RoPE 并打
  warning，避免在未验证的近似上静默产出错误数值。

张量形状约定
------------
- ``q`` / ``k``: ``[batch, num_heads, seq_len, head_dim]``；
- ``cos`` / ``sin``: ``[batch, 1, seq_len, head_dim]``（广播到多头）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.model.config import ModelConfig

logger = get_logger("llama31_8b_scratch.model.rope")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """对半拆分后交换并取负，对应 RoPE 公式中的 ``rotate_half``。

    Args:
        x: 最后一维长度为 ``head_dim`` 的张量。

    Returns:
        形状与 ``x`` 相同的张量，前半段为 ``-x[..., d/2:]``，
        后半段为 ``x[..., :d/2]``。
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _build_inv_freq(head_dim: int, rope_theta: float) -> torch.Tensor:
    """构造基础逆频率向量 ``[head_dim/2]``。

    公式：:math:`\\omega_i = \\theta_{\\text{base}}^{-2i/d}`。
    """
    index = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (index / head_dim))
    return inv_freq


class ScratchRotaryEmbedding(nn.Module):
    """为注意力 Q/K 提供 cos/sin 相位表的旋转位置编码模块。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling: Optional[Dict[str, Any]] = config.rope_scaling

        inv_freq = _build_inv_freq(self.head_dim, self.rope_theta)
        self._apply_scaling(inv_freq)

        # persistent=False：inv_freq 由配置确定性重建，不写入 state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------
    # scaling 分支
    # ------------------------------------------------------------------
    def _apply_scaling(self, inv_freq: torch.Tensor) -> None:
        """按 ``rope_scaling`` 原地修正逆频率（linear / dynamic / yarn）。"""
        if not self.rope_scaling:
            return

        scaling_type = self.rope_scaling.get("type") or self.rope_scaling.get(
            "rope_type"
        )
        factor = float(self.rope_scaling.get("factor", 1.0))
        dim = self.head_dim

        if scaling_type == "linear":
            # 频率整体除以 s：位置轴被线性拉长 s 倍
            inv_freq.div_(factor)
        elif scaling_type in ("dynamic", "ntk"):
            # NTK-by-parts：inv_freq_i *= s^(2i/d)
            idx = torch.arange(0, dim, 2, dtype=torch.float32)
            inv_freq.mul_(factor ** (idx / dim))
        elif scaling_type == "yarn":
            # 完整 YaRN 数学在 README 讲解；代码侧先回退基础 RoPE。
            logger.warning(
                "rope_scaling.type='yarn' 尚未在代码侧完整实现，已回退基础 RoPE；"
                "完整推导见 02_model/README.md。"
            )
        else:
            logger.warning("未知 rope_scaling 类型 %r，按基础 RoPE 处理", scaling_type)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算给定位置的 cos / sin 相位表。

        Args:
            x: 注意力 Q（或 K），形状 ``[B, H, T, D]``，仅用于取 dtype/device。
            position_ids: 位置编号，形状 ``[B, T]``。

        Returns:
            ``(cos, sin)``，形状均为 ``[B, 1, T, D]``。
        """
        # inv_freq: [D/2] -> [1, D/2, 1]；position_ids: [B, T] -> [B, 1, T]
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        position_expanded = position_ids[:, None, :].float()

        # freqs[b, i, t] = position[b, t] * inv_freq[i]，形状 [B, D/2, T]
        freqs = inv_freq_expanded @ position_expanded
        # 转成 [B, T, D/2]，再复制成 [B, T, D]（前半/后半相位相同）
        freqs = freqs.transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype=x.dtype)[:, None, :, :]
        sin = emb.sin().to(dtype=x.dtype)[:, None, :, :]
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 相位作用到 Q、K 上。

    Args:
        q: ``[B, Hq, T, D]``；k: ``[B, Hkv, T, D]``。
        cos, sin: ``[B, 1, T, D]``。

    Returns:
        旋转后的 ``(q_embed, k_embed)``，形状与输入一致。
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 供外部计算相位时复用的工具函数（纯数学，无模块状态）
def build_default_rope_parameters(
    head_dim: int, rope_theta: float
) -> Tuple[torch.Tensor, float]:
    """返回基础 ``inv_freq`` 与注意力缩放因子 1.0（教学工具函数）。"""
    return _build_inv_freq(head_dim, rope_theta), 1.0




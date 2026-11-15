r"""RMSNorm 归一化层（手写实现）。

数学定义
--------
对最后一维长度为 :math:`d` 的向量 :math:`x \in \mathbb{R}^{d}`，先计算均方根：

.. math::

    \mathrm{RMS}(x) = \sqrt{\\frac{1}{d}\sum_{i=1}^{d} x_i^{2} + \epsilon}

再做逐维缩放并施加可学习权重 :math:`w \in \mathbb{R}^{d}`：

.. math::

    \hat{x}_i = \\frac{x_i}{\mathrm{RMS}(x)} \cdot w_i

与 LayerNorm 的区别：RMSNorm 不再减去均值、也没有偏置项，计算量更小，
Llama / Qwen / Mistral 等现代解码器模型统一采用 RMSNorm。

张量形状约定
------------
- 输入 ``hidden_states``: ``[batch, seq_len, hidden_size]``（也兼容更高维前缀）；
- 输出形状与输入完全一致；
- 参数 ``weight``: ``[hidden_size]``，初始化为全 1。
"""

from __future__ import annotations

import torch
from torch import nn


class ScratchRMSNorm(nn.Module):
    """RMSNorm，权重键名与 HF Llama 的 ``input_layernorm.weight`` 等逐字对应。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        # weight 是逐通道缩放系数；nn.Parameter 会被纳入 state_dict，
        # 键名由外层模块的属性路径决定（如 model.layers.0.input_layernorm.weight）
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """对最后一维做 RMS 归一化。

        Args:
            hidden_states: 任意前缀形状，最后一维必须等于 ``hidden_size``。

        Returns:
            归一化并乘以 ``weight`` 后的张量，形状与输入一致。
        """
        input_dtype = hidden_states.dtype
        # 方差在 fp32 下计算，避免半精度时 x^2 下溢导致 RMS 偏差
        hidden_states = hidden_states.to(torch.float32)

        # variance = mean(x^2, dim=-1, keepdim=True)，形状 [..., 1]
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt(variance + eps) 即 1 / RMS(x)；等价于公式中的 x / RMS(x)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 乘回可学习权重，并恢复原始 dtype（半精度训练时避免类型反复转换）
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"

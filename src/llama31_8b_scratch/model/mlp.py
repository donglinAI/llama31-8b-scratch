r"""SwiGLU 前馈网络（FFN）手写实现。

数学定义
--------
标准 Transformer FFN 为 :math:`\\mathrm{FFN}(x) = W_2 \\sigma(W_1 x)`；
Llama 系列采用 SwiGLU 门控形式：

.. math::

    \\mathrm{SwiGLU}(x) = W_{\\text{down}} \\big(
        \\sigma(W_{\\text{gate}} x) \\odot W_{\\text{up}} x
    \\big)

其中 :math:`\\sigma` 为 SiLU（Swish）激活：

.. math::

    \\sigma(z) = z \\cdot \\mathrm{sigmoid}(z)

张量形状
--------
- 输入 ``x``: ``[B, T, H]``；
- ``gate_proj`` / ``up_proj``: ``H -> intermediate``；
- ``down_proj``: ``intermediate -> H``；
- 输出 ``[B, T, H]``。

权重键名（与 HF Llama 逐字一致）：
``mlp.gate_proj.weight``、``mlp.up_proj.weight``、``mlp.down_proj.weight``。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llama31_8b_scratch.model.config import ModelConfig

#: 激活函数注册表；当前模型配置只允许 silu，其余保留给教学对照
_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
}


class ScratchMLP(nn.Module):
    """SwiGLU 两层前馈网络。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Llama 系列 MLP 投影层无 bias（与 attention_bias 解耦）
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act not in _ACT2FN:
            raise ValueError(f"MLP 不支持的激活函数: {config.hidden_act!r}")
        self.act_fn = _ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前向：门控分支 × 上行分支，再投影回隐藏维。"""
        # gate: [B, T, I]；up: [B, T, I]
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        # 逐元素相乘后 down 投影：[B, T, I] -> [B, T, H]
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, act={self.act_fn.__name__}"
        )

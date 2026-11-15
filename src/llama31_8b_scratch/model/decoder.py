r"""解码器层（ScratchDecoderLayer）手写实现。

结构（pre-norm 残差块，Llama 口径）
----------------------------------
.. code-block:: text

    x ──► input_layernorm ──► self_attn ──► + ◄── residual
      │                                      │
      └──────────────────────────────────────┘
      ──► post_attention_layernorm ──► mlp ──► + ◄── residual
                                                │
                                              输出

数学上每个子层为：

.. math::

    x' = x + \\mathrm{Attn}(\\mathrm{RMSNorm}(x))
    x'' = x' + \\mathrm{MLP}(\\mathrm{RMSNorm}(x'))

梯度检查点（gradient checkpointing）
------------------------------------
训练显存紧张时，子层中间激活不保存、反向传播时重算前向；本模块通过
``torch.utils.checkpoint.checkpoint`` 惰性包裹，只在 ``training=True`` 且
显式开启时生效，推理路径不受影响。

子模块权重键名（与 HF Llama 逐字一致）：
``input_layernorm.weight``、``post_attention_layernorm.weight``、
``self_attn.*``、``mlp.*``。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from llama31_8b_scratch.model.attention import ScratchAttention
from llama31_8b_scratch.model.config import ModelConfig
from llama31_8b_scratch.model.mlp import ScratchMLP
from llama31_8b_scratch.model.norm import ScratchRMSNorm


class ScratchDecoderLayer(nn.Module):
    """单层 Transformer 解码器块。"""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ScratchAttention(config, layer_idx=layer_idx)
        self.mlp = ScratchMLP(config)
        self.input_layernorm = ScratchRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ScratchRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """前向一层。

        Args:
            hidden_states: ``[B, T, H]``。
            position_ids: ``[B, T]``。
            attention_mask: 加性因果 mask，``[B, 1, Tq, Tk]``。
            past_key_value: KV 缓存实例。
            use_cache: 是否更新缓存。
            gradient_checkpointing: 是否对本层启用梯度检查点。

        Returns:
            ``(hidden_states, attn_weights, past_key_value)``。
        """
        if gradient_checkpointing and self.training and torch.is_grad_enabled():
            return self._checkpointed_forward(
                hidden_states, position_ids, attention_mask,
                past_key_value, use_cache,
            )
        return self._forward_impl(
            hidden_states, position_ids, attention_mask,
            past_key_value, use_cache,
        )

    # ------------------------------------------------------------------
    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """用 ``torch.utils.checkpoint`` 包裹前向，换取训练显存下降。"""

        def _run(*inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            h, pos = inputs[0], inputs[1]
            out = self._forward_impl(h, pos, attention_mask, past_key_value, use_cache)
            return out[0], out[2]

        hidden_states, past_key_value = torch_checkpoint.checkpoint(
            _run, hidden_states, position_ids, use_reentrant=False,
        )
        return hidden_states, None, past_key_value

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        # 子层 1：注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, past_key_value = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 子层 2：MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_key_value

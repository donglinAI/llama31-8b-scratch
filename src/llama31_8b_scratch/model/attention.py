r"""多头 / 分组查询注意力（MHA / GQA）手写实现。

结构概览
--------
1. 输入 ``hidden_states``: ``[B, T, H]`` 经 Q/K/V 三个线性投影，重塑为多头：
   - Q: ``[B, num_heads, T, head_dim]``
   - K/V: ``[B, num_kv_heads, T, head_dim]``（GQA 下 KV 头数更少）
2. 对 Q/K 施加 RoPE 位置编码；
3. 若启用 KV Cache，把当前步 K/V 拼到缓存尾部；
4. 用手写 :func:`repeat_kv` 把 KV 头复制到与 Q 头一一对应（GQA 分组）；
5. 按配置选择 ``eager`` / ``sdpa`` / ``flash_attention_2`` 计算注意力；
6. 输出头合并后经 ``o_proj`` 投影回 ``hidden_size``。

GQA 分组数学
------------
设 Q 头数 :math:`n_h`，KV 头数 :math:`n_{kv}`，分组大小
:math:`g = n_h / n_{kv}`。每个 KV 头服务于 :math:`g` 个 Q 头，
复制操作等价于在 KV 头维度上 ``repeat`` 后重排：

.. math::

    K_{\\text{rep}}[b, h, t, d] = K[b, h // g, t, d]

三种实现
--------
- ``eager``：手写 ``QK^T / sqrt(d)`` → 加因果 mask → softmax → 乘 V，教学用；
- ``sdpa``：调用 ``F.scaled_dot_product_attention``，内部自动选择高效内核；
- ``flash_attention_2``：惰性导入 ``flash_attn``；本机未安装时打印 warning
  并回退到 ``sdpa``，保证纯 CPU 环境 import 与前向不报错。

投影权重键名（与 HF Llama 逐字一致）：
``self_attn.q_proj.weight``、``self_attn.k_proj.weight``、
``self_attn.v_proj.weight``、``self_attn.o_proj.weight``。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.model.config import ModelConfig
from llama31_8b_scratch.model.rope import ScratchRotaryEmbedding, apply_rotary_pos_emb

logger = get_logger("llama31_8b_scratch.model.attention")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头按 GQA 分组复制到 Q 头数量。

    Args:
        hidden_states: ``[B, num_kv_heads, T, head_dim]``。
        n_rep: 每个 KV 头需要复制的份数（``num_heads // num_kv_heads``）。

    Returns:
        ``[B, num_kv_heads * n_rep, T, head_dim]``。当 ``n_rep == 1`` 时原样返回。
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 先在 KV 头后插入复制维度，expand 后合并回头维度
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class ScratchAttention(nn.Module):
    """解码器单层自注意力。"""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        # 投影层：bias 由配置 attention_bias 决定（Llama 3.1 默认无 bias）
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias,
        )

        self.rotary_emb = ScratchRotaryEmbedding(config)
        self.attention_dropout = config.attention_dropout

        # 注意力实现选择：eager / sdpa / flash_attention_2
        self._attn_implementation = self._resolve_impl(config)

    # ------------------------------------------------------------------
    # 实现选择
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_impl(config: ModelConfig) -> str:
        """从配置读取注意力实现，非法值回退 sdpa。"""
        impl = getattr(config, "attention_impl", None) or "sdpa"
        if impl not in ("eager", "sdpa", "flash_attention_2"):
            logger.warning("未知注意力实现 %r，回退 sdpa", impl)
            return "sdpa"
        return impl

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """计算自注意力输出。

        Args:
            hidden_states: ``[B, T, H]``。
            position_ids: ``[B, T]``。
            attention_mask: 加性 mask，形状 ``[B, 1, Tq, Tk]``，被屏蔽位置填
                ``-inf``；``None`` 表示不额外屏蔽。
            past_key_value: :class:`llama31_8b_scratch.generation.cache.KVCache` 实例。
            use_cache: 是否在返回时更新 KV 缓存。

        Returns:
            ``(attn_output, attn_weights, past_key_value)``；``attn_weights``
            在 sdpa/flash 路径下不计算，返回 ``None``。
        """
        bsz, q_len, _ = hidden_states.shape

        # --- Q/K/V 投影并重塑为多头 ---
        # q: [B, T, n_h*D] -> [B, n_h, T, D]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # --- RoPE 位置编码 ---
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- KV Cache：把当前 K/V 拼到缓存 ---
        if past_key_value is not None and use_cache:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # --- GQA：KV 头复制到 Q 头数量 ---
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # --- 分派到具体注意力内核 ---
        if self._attn_implementation == "eager":
            attn_output, attn_weights = self._eager(q, k, v, attention_mask)
        elif self._attn_implementation == "flash_attention_2":
            attn_output, attn_weights = self._flash_with_fallback(q, k, v, attention_mask)
        else:
            attn_output, attn_weights = self._sdpa(q, k, v, attention_mask)

        # --- 合并多头：[B, n_h, T, D] -> [B, T, n_h*D] -> o_proj ---
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    # ------------------------------------------------------------------
    # 三种内核
    # ------------------------------------------------------------------
    def _eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """手写缩放点积注意力：教学路径，便于看清每一步形状。"""
        # scores = Q K^T / sqrt(d)，形状 [B, n_h, Tq, Tk]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        bsz, _, q_len, kv_len = attn_weights.shape
        if attention_mask is None and q_len > 1:
            # 无显式 mask 时内部补因果三角掩码：未来位置置 -inf
            causal = torch.tril(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
            )
            attn_weights = attn_weights.masked_fill(~causal, float("-inf"))
        elif attention_mask is not None:
            # 加性 mask：被屏蔽位置为 -inf，softmax 后权重趋零
            attn_weights = attn_weights + attention_mask

        # softmax 在 fp32 下计算，再回落到 q 的 dtype
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch 内置 SDPA（底层可路由到 efficient / math / flash 内核）。"""
        dropout_p = self.attention_dropout if self.training else 0.0
        # is_causal 仅在无显式 mask 且查询长度大于 1 时开启；
        # 解码步 q_len=1 不需要因果三角 mask。
        is_causal = attention_mask is None and q.shape[2] > 1
        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return attn_output, None

    def _flash_with_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FlashAttention-2 路径；``flash_attn`` 缺失时回退 sdpa。

        注意：``flash_attn`` 属于可选重依赖，按 SPEC 第 3 节要求惰性导入，
        禁止出现在模块顶层。
        """
        try:
            from flash_attn import flash_attn_func  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "未安装 flash_attn，flash_attention_2 回退到 sdpa；"
                "如需 FA2 内核请在 GPU 环境安装 flash-attn。"
            )
            return self._sdpa(q, k, v, attention_mask)

        # flash_attn_func 期望 [B, T, H, D]，与现有 [B, H, T, D] 互换
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=attention_mask is None and q.shape[1] > 1,
        )
        attn_output = attn_output.transpose(1, 2)
        return attn_output, None

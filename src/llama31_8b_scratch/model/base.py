"""解码器骨干模型 ScratchModel 手写实现。

职责
----
- 词元嵌入查表；
- 逐层堆叠 :class: ScratchDecoderLayer;
- 最终 RMSNorm;
- 构造因果加性 mask, 同时支持 padding mask 与 KV Cache 解码步;

state_dict 顶层键路径约定： model.embed_tokens.weight、
model.layers.{i}.*、model.norm.weight, 与 HF Llama 对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from llama31_8b_scratch.model.config import ModelConfig
from llama31_8b_scratch.model.decoder import ScratchDecoderLayer
from llama31_8b_scratch.model.embeddings import ScratchEmbedding
from llama31_8b_scratch.model.norm import ScratchRMSNorm


@dataclass
class BaseModelOutput:
    """骨干前向输出的轻量容器，避免依赖 transformers 的输出类。"""

    last_hidden_state: torch.Tensor # Decoder 最后一层的输出特征
    past_key_values: Optional[Any] = None # KV 缓存，推理时用；训练阶段一般传 None 
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None # 全部层的中间隐状态（可视化、蒸馏、分析）的时候才填充，正常训练推理可不传


class ScratchModel(nn.Module):
    """纯 nn.Module 解码器骨干，不继承任何 HF 预训练模型类。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__() 
        self.config = config
        self.embed_tokens = ScratchEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )
        self.layers = nn.ModuleList(
            [ScratchDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        ) # ModuleList 只是容器，不会自动串联；forward 里要写循环依次过每一层
        self.norm = ScratchRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Decoder计算完后，进入 lm_head 之前的一次 全局 RMSNorm

        # 权重初始化（Llama 口径：线性层 normal_(std=initializer_range)）
        self.apply(self._init_weights) # 遍历模型的所有子模块，逐个执行 _init_weights 函数

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        """按配置的 initializer_range 初始化线性层与嵌入层。"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ScratchEmbedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # mask 构造
    # ------------------------------------------------------------------
    def _prepare_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        past_key_values: Optional[Any],
    ) -> Optional[torch.Tensor]:
        """构造加性因果 mask, 形状 [B, 1, Tq, Tk]，屏蔽位置填 -inf 

        Args:
            attention_mask: [B, T], 1 表示有效 token、0 表示 padding; 可空。
            input_ids: 当前步输入，[B, Tq]。
            past_key_values: KV 缓存，用于推算历史键长度 past_len
        """

        bsz, q_len = input_ids.shape
        past_len = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_len = past_key_values.get_seq_length()
        kv_len = past_len + q_len

        # 因果三角：位置 t 只能看到键位置 <= t 的位置
        causal = torch.tril(
            torch.ones(q_len, kv_len, dtype=torch.bool, device=input_ids.device)
        )
        # 加性 mask：允许位置为 0，屏蔽位置为 -inf
        additive = torch.zeros(
            q_len, kv_len, dtype=torch.float32, device=input_ids.device
        )
        additive = additive.masked_fill(~causal, float("-inf"))

        if attention_mask is not None:
            # padding mask: [B, T] -> [B, 1, 1, Tk]，被 pad 的键位置 -inf
            pad_mask = attention_mask[:, None, None, :].to(torch.bool)
            pad_additive = torch.zeros(
                bsz, 1, q_len, kv_len, dtype=torch.float32, device=input_ids.device
            )
            pad_additive = pad_additive.masked_fill(~pad_mask, float("-inf"))
            # 因果 mask 扩成 [1,1,q,kv] 与 [B,1,q,kv] 的 padding mask 广播相加
            additive = additive[None, None, :, :] + pad_additive

        return additive

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """骨干前向传播。

        Args:
            input_ids: ``[B, T]``。
            position_ids: ``[B, T]``，缺省按顺序编号。
            attention_mask: padding 掩码，``[B, T]``。
            past_key_values: KV 缓存实例。
            use_cache: 是否在层间更新缓存。
            gradient_checkpointing: 训练时是否启用梯度检查点。
            output_hidden_states: 是否收集每层隐藏状态（调试/评测用）。
        """
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(bsz, -1)

        causal_mask = self._prepare_causal_mask(attention_mask, input_ids, past_key_values)

        all_hidden: Optional[List[torch.Tensor]] = [] if output_hidden_states else None

        for layer in self.layers:
            if all_hidden is not None:
                all_hidden.append(hidden_states)
            hidden_states, _, past_key_values = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                use_cache=use_cache,
                gradient_checkpointing=gradient_checkpointing,
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden is not None:
            all_hidden.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden) if all_hidden is not None else None,
        )

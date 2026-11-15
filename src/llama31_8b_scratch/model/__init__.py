"""手写 Transformer 解码器模型包（HF Llama 权重兼容）。

对外暴露的核心类：

- :class:`ModelConfig`：模型超参数配置；
- :class:`ScratchRMSNorm`：RMS 归一化；
- :class:`ScratchRotaryEmbedding`：旋转位置编码；
- :class:`ScratchAttention`：MHA / GQA 自注意力；
- :class:`ScratchMLP`：SwiGLU 前馈网络；
- :class:`ScratchDecoderLayer`：解码器层；
- :class:`ScratchModel`：骨干模型；
- :class:`ScratchForCausalLM`：因果语言模型；
- :class:`ScratchForSequenceClassification`：奖励 / 序列分类头。
"""

from __future__ import annotations

from llama31_8b_scratch.model.attention import ScratchAttention, repeat_kv
from llama31_8b_scratch.model.base import BaseModelOutput, ScratchModel
from llama31_8b_scratch.model.causal_lm import CausalLMOutput, ScratchForCausalLM
from llama31_8b_scratch.model.config import ModelConfig
from llama31_8b_scratch.model.decoder import ScratchDecoderLayer
from llama31_8b_scratch.model.embeddings import ScratchEmbedding
from llama31_8b_scratch.model.mlp import ScratchMLP
from llama31_8b_scratch.model.norm import ScratchRMSNorm
from llama31_8b_scratch.model.rope import (
    ScratchRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from llama31_8b_scratch.model.sequence import ScratchForSequenceClassification

__all__ = [
    "ModelConfig",
    "ScratchRMSNorm",
    "ScratchRotaryEmbedding",
    "ScratchEmbedding",
    "ScratchAttention",
    "repeat_kv",
    "ScratchMLP",
    "ScratchDecoderLayer",
    "ScratchModel",
    "BaseModelOutput",
    "ScratchForCausalLM",
    "CausalLMOutput",
    "ScratchForSequenceClassification",
    "apply_rotary_pos_emb",
    "rotate_half",
]

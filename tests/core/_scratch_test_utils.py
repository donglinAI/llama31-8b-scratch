"""模型单元测试共享工具：构造 CPU 可跑的极小配置与模型。

本文件不匹配 ``model_*.py`` 发现规则，不会被 unittest 当作测试用例。
"""

from __future__ import annotations

from typing import Any, Dict

from llama31_8b_scratch.model.causal_lm import ScratchForCausalLM
from llama31_8b_scratch.model.config import ModelConfig


def smoke_config(**overrides: Any) -> ModelConfig:
    """返回 smoke 同构的极小模型配置，可用关键字参数覆盖字段。"""
    base: Dict[str, Any] = dict(
        arch="scratch",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        initializer_range=0.02,
        use_cache=True,
        rope_scaling=None,
        token_ids={
            "bos_token_id": 1,
            "eos_token_id": 2,
            "eot_token_id": 3,
            "pad_token_id": 0,
        },
    )
    base.update(overrides)
    return ModelConfig(**base)


def build_causal_lm(**overrides: Any) -> ScratchForCausalLM:
    """按极小配置构建一个因果 LM（eval 模式由测试自行切换）。"""
    return ScratchForCausalLM(smoke_config(**overrides))

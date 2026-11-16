"""模型配置数据类。

本模块只负责把 YAML 中的 model / token 节点翻译成强类型的
:class:ModelConfig, 不实例化任何 nn.Module。

to_hf_dict 产出的字典可直接写成标准 HuggingFace config.json (model_type="llama"), 
供 transformers / vLLM / SGLang 以原生 Llama 结构加载，无需注册自定义模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llama31_8b_scratch.common.config import Config


#: 配置中允许的激活函数名（Llama 系列默认 silu，SwiGLU 内部使用）
_SUPPORTED_ACTS = ("silu", "gelu", "gelu_pytorch_tanh")


@dataclass # 自动生成 __init__ __repr__ 等方法, 省去大量赋值代码
class ModelConfig: # 模型超参数类
    """手写 Transformer 的超参数配置。

    属性与 configs/model/*.yaml 的 model 节点一一对应; token_ids
    是从 token 节点取出的特殊 token id 字典，供生成与对齐逻辑使用，
    不属于模型结构超参数。
    """

    # ---- 结构超参数（字段名不得改动） ----
    arch: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    tie_word_embeddings: bool # 是否共享输入 embedding 和 输出 lm_head 权重；
    initializer_range: float
    use_cache: bool # 是否开启 KV 缓存，训练阶段关闭，推理阶段开启，加速自回归
    rope_scaling: Optional[Dict[str, Any]] = field(default=None) # RoPE缩放配置
    # 注意力实现选择：eager / sdpa / flash_attention_2（可选，默认 sdpa）
    attention_impl: str = "sdpa"

    # ---- 特殊 token id（来自 YAML 的 token 节点） ----
    token_ids: Dict[str, int] = field(default_factory=dict) # 必须用 default_factory=dict 不能用 {}, 需要单独生成一个全新空字典；否则会让不同实例共享字段

    # ------------------------------------------------------------------
    # 构造与校验
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml_config(cls, cfg: Config) -> "ModelConfig":
        """从公共配置对象构建 ModelConfig ; 

        Args:
            cfg: llama31_8b_scratch.common.config.Config
        
        """

        if "model" not in cfg:
            raise ValueError("配置中缺少 model 节点，无法构建 ModelConfig")

        model = cfg.model
        token_node = cfg.get("token", None)

        token_ids: Dict[str, int] = {}
        if token_node is not None:
            # Config.get 返回原始 dict；统一成属性/字典两种读取口径
            def _get(key: str) -> Any:
                if isinstance(token_node, dict):
                    return token_node.get(key)
                return getattr(token_node, key, None)

            token_ids = {
                "bos_token_id": _get("bos_token_id"),
                "eos_token_id": _get("eos_token_id"),
                "eot_token_id": _get("eot_token_id"),
                "pad_token_id": _get("pad_token_id"),
                "start_header_id": _get("start_header_id"),
                "end_header_id": _get("end_header_id"),
            }
            token_ids = {k: v for k, v in token_ids.items() if v is not None}

        config = cls(
            arch=str(model.arch),
            vocab_size=int(model.vocab_size),
            hidden_size=int(model.hidden_size),
            intermediate_size=int(model.intermediate_size),
            num_hidden_layers=int(model.num_hidden_layers),
            num_attention_heads=int(model.num_attention_heads),
            num_key_value_heads=int(model.num_key_value_heads),
            head_dim=int(model.head_dim),
            max_position_embeddings=int(model.max_position_embeddings),
            rope_theta=float(model.rope_theta),
            rms_norm_eps=float(model.rms_norm_eps),
            hidden_act=str(model.hidden_act),
            attention_bias=bool(model.attention_bias),
            attention_dropout=float(model.attention_dropout),
            tie_word_embeddings=bool(model.tie_word_embeddings),
            initializer_range=float(model.initializer_range),
            use_cache=bool(model.use_cache),
            rope_scaling=model.get("rope_scaling", None)
            if hasattr(model, "get")
            else getattr(model, "rope_scaling", None),
            attention_impl=str(model.get("attention_impl", "sdpa"))
            if hasattr(model, "get")
            else getattr(model, "attention_impl", "sdpa"),
            token_ids=token_ids,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """做结构一致性校验，提前暴露配置错误而不是在矩阵乘法处崩溃。"""
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads({self.num_attention_heads}) 必须能被 "
                f"num_key_value_heads({self.num_key_value_heads}) 整除（GQA 分组数）"
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 应等于 "
                f"num_attention_heads({self.num_attention_heads}) × head_dim({self.head_dim})"
            )
        if self.hidden_act not in _SUPPORTED_ACTS:
            raise ValueError(f"不支持的激活函数: {self.hidden_act!r}")
        if self.intermediate_size <= 0 or self.vocab_size <= 0:
            raise ValueError("intermediate_size / vocab_size 必须为正整数")

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def num_kv_groups(self) -> int:
        """每个 KV 头对应多少个 Q 头（GQA 分组大小）。"""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def eos_token_id(self) -> Optional[int]:
        """默认结束 token id（``eos_token_id``，对齐层可改用 ``eot_token_id``）。"""
        return self.token_ids.get("eos_token_id")

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.token_ids.get("pad_token_id")

    # ------------------------------------------------------------------
    # HF 兼容导出
    # ------------------------------------------------------------------
    def to_hf_dict(self) -> Dict[str, Any]:
        """转成可写入 ``config.json`` 的字典（标准 Llama 结构）。

        说明：
        - ``model_type`` 固定为 ``"llama"``，下游框架按原生 Llama 加载；
        - 同时保留经典字段 ``rope_theta`` 与 ``rope_scaling``，兼容
          transformers 4.x / 5.x 以及 vLLM、SGLang 的读取路径；
        - 不写 ``auto_map``，因为权重键名与原生 Llama 逐字一致。
        """
        hf: Dict[str, Any] = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "tie_word_embeddings": self.tie_word_embeddings,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "rope_scaling": self.rope_scaling,
        }
        for key in ("bos_token_id", "eos_token_id", "eot_token_id", "pad_token_id"):
            if key in self.token_ids:
                hf[key] = self.token_ids[key]
        return hf

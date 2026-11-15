"""LoRA / QLoRA（NF4）封装：惰性 import peft / bitsandbytes。

一、LoRA 原理（为什么只训 <1% 参数就能微调好）

LoRA（Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"）的核心假设：
大模型在适配下游任务时，权重矩阵的"改动量" ΔW 具有低秩。于是把 ΔW 分解为两个小矩阵：

.. math::
    W' = W_0 + \\Delta W = W_0 + B A,\\quad
    B \\in \\mathbb{R}^{d\\times r},\\ A \\in \\mathbb{R}^{r\\times k},\\ r \\ll \\min(d,k)

推理时可以把 ``BA`` 合并回 ``W_0``，零延迟；训练时只更新 A、B，冻结 W_0。
参数量从 ``d*k`` 降到 ``r*(d+k)``，r=8 时通常是原权重的 0.1%~1%。

二、QLoRA / NF4（Dettmers et al., 2023）

QLoRA 在 LoRA 之前再加一步：把 base 模型用 4-bit 归一化浮点（NF4）量化冻结，
只把 LoRA 的 A、B 保持 bf16 训练。这样 33B 模型可以在单张 24GB 卡上微调。
NF4 是为正态分布权重设计的 4-bit 量化格式，比普通 INT4 更接近原始分布。

三、为什么本模块要"惰性"且要"降级分支"

- ``peft``、``bitsandbytes`` 都是重型、且 bnb 只在 CUDA Linux 可用；
- CPU smoke / 单测环境不装这些库，但仍要能 import 本模块、并验证"冻结 base 参数"
  这条分支逻辑；
- 因此 ``apply_lora`` 在 peft 缺失时**不会崩**：它会把所有 base 参数 ``requires_grad=False``，
  并通过返回值告知调用方"走的是无 peft 降级分支"。真要训练时若没有可训参数，
  训练循环会自然报错，行为可预期。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["LoraConfig", "apply_lora", "count_trainable_parameters", "build_bnb_quant_config"]


@dataclass
class LoraConfig:
    """LoRA 超参（与 peft.LoraConfig 字段对齐）。"""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    #: 要注入 LoRA 的模块名子串，例如 ["q_proj", "v_proj"]
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # none / all / lora_only
    task_type: str = "CAUSAL_LM"
    #: 是否启用 QLoRA（NF4 量化 base 模型）
    use_qlora: bool = False
    #: QLoRA 4-bit 量化位宽（4 或 8）
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """返回 ``{"trainable": int, "total": int, "ratio": float}``。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / max(1, total),
    }


def _freeze_base(model: nn.Module) -> None:
    """把所有参数 requires_grad 置 False（peft 缺失时的降级分支）。"""
    for p in model.parameters():
        p.requires_grad = False


def apply_lora(
    model: nn.Module, cfg: LoraConfig
) -> tuple[nn.Module, str]:
    """给模型注入 LoRA。

    Returns:
        ``(model, backend)``：``backend`` 取值
        - ``"peft"``：真正用 peft 注入了 LoRA 适配器；
        - ``"manual_freeze"``：peft 不可用，已把 base 参数全部冻结（仅用于测试 / dry-run）。

    Note:
        当 ``use_qlora=True`` 时，本函数假定模型已经被 ``bitsandbytes`` 量化过（通常
        在 HF 加载路径里做）。对纯手写 ``ScratchForCausalLM``，QLoRA 量化需要外部先把
       权重替换成 4-bit 线性层，本模块只负责注入 LoRA A/B。
    """
    # 先尝试 peft 路径
    try:
        from peft import LoraConfig as PeftLoraConfig  # type: ignore
        from peft import get_peft_model  # type: ignore
    except Exception as exc:
        logger.warning(
            "peft 不可用（%s），走 manual_freeze 降级分支：base 参数已全部冻结；"
            "如需真正 LoRA 训练请 pip install peft",
            exc,
        )
        _freeze_base(model)
        return model, "manual_freeze"

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )
    model = get_peft_model(model, peft_cfg)
    stats = count_trainable_parameters(model)
    logger.info(
        "LoRA 注入完成（peft）：可训练参数 %d / %d = %.4f%%",
        stats["trainable"], stats["total"], stats["ratio"] * 100,
    )
    return model, "peft"


def build_bnb_quant_config(cfg: LoraConfig) -> Optional[Dict[str, Any]]:
    """构造给 ``transformers.BitsAndBytesConfig`` 用的字典（QLoRA NF4）。

    惰性 import：调用方在加载 HF 模型时把这个 dict 传进去。纯手写 Scratch 模型不直接用。
    """
    if not cfg.use_qlora:
        return None
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        logger.warning("bitsandbytes 不可用，QLoRA 量化配置将无法生效：%s", exc)
        return None

    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,  # "nf4"
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
    }

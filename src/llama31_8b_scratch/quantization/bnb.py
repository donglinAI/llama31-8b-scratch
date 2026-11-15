"""bitsandbytes 8bit / 4bit(NF4) 量化（惰性封装）。

bitsandbytes 是什么
===================
`bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`_ 由 Tim Dettmers
等人主导，是目前 HF transformers / PEFT / TRL 生态里**事实标准**的
量化后端。它和 GPTQ/AWQ 思路不同：

- **GPTQ / AWQ**：离线把权重量化好、存成新 checkpoint，推理时直接读。
- **bitsandbytes**：**加载时动态量化**——模型权重从 safetensors 读出来后，
  在 GPU 上实时反量化成 int8/int4，不重训、不校准（4bit NF4 也是纯数学映射）。
  所以它天生适合"加载别人的模型来做 QLoRA 微调"。

8bit 混合精度分解
=================
大模型权重里存在少量 **outlier 通道**（某些维度绝对值远大于其它）。如果把
整个矩阵按最大绝对值做 per-channel absmax 量化，outlier 会把 scale 撑大，
其它正常维度的有效精度被压没。bnb 8bit 的做法是：

1. 把权重矩阵拆成两块——绝大多数普通维度走 int8，少数 outlier 维度留 fp16；
2. 用一个"双量化常数"在推理时把两块拼回去。

这就是 *LLM.int8()*（Dettmers et al., 2022）的核心。

4bit NF4（QLoRA 的基石）
========================
NF4（NormalFloat 4-bit，Dettmers 等人在 QLoRA 论文 2023 中提出）不是均匀
网格。它的 16 个量化级**按照标准正态分布的分位点非均匀布置**：

.. math:
    q_i = \\Phi^{-1}\\!\\left(\\frac{i}{15}\\right), \\quad i=0,\\dots,15

为什么这样做？因为大量实验观察到 LLMs 的权重在归一化后近似服从标准正态，
**分布在 0 附近的权重大头需要精细网格，分布在两端的权重稀少可以稀疏些**。
均匀 int4 在 0 附近的精度反而不如 NF4。

**double quant**：连第一步算出来的 fp32 scale 本身也再做一次 int8 量化，
省下来的显存大约每参数 0.5 bit。

**paged optimizer**：QLoRA 训练时，AdamW 的 optimizer state 用 NVIDIA
统一内存分页，峰值显存被打爆时自动 offload 到 CPU，避免 OOM。这是一个
训练侧特性，不是推理侧量化。

调用方式（惰性）
================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["build_4bit_config", "build_8bit_config", "load_quantized_model"]


def build_4bit_config(
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    use_double_quant: bool = True,
):
    """构造 transformers 用的 4bit BitsAndBytesConfig。

    Args:
        compute_dtype: 计算时的精度（``bfloat16`` / ``float16`` / ``float32``）。
            权重仍是 4bit 存储，反量化后转成这个 dtype 参与 GEMM。
        quant_type: ``"nf4"``（推荐，正态分布网格）或 ``"fp4"``（均匀网格）。
        use_double_quant: 是否对 scale 再做一次量化（省显存）。
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 transformers：pip install transformers accelerate"
        ) from exc

    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if compute_dtype not in dtype_map:
        raise ValueError(f"compute_dtype 必须是 {list(dtype_map)}")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=dtype_map[compute_dtype],
    )


def build_8bit_config():
    """构造 transformers 用的 8bit BitsAndBytesConfig（LLM.int8()）。"""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 transformers：pip install transformers accelerate"
        ) from exc

    return BitsAndBytesConfig(load_in_8bit=True)


def load_quantized_model(
    model_path: str | Path,
    *,
    mode: str = "4bit",
    compute_dtype: str = "bfloat16",
):
    """用 bnb 动态量化加载 HF 模型。

    Args:
        model_path: HF 模型目录。
        mode: ``"4bit"`` 或 ``"8bit"``。
        compute_dtype: 4bit 计算精度。

    Returns:
        ``AutoModelForCausalLM.from_pretrained(...)`` 返回的模型。
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 transformers：pip install transformers accelerate"
        ) from exc

    if mode == "4bit":
        bnb_config = build_4bit_config(compute_dtype=compute_dtype)
    elif mode == "8bit":
        bnb_config = build_8bit_config()
    else:
        raise ValueError(f"mode 必须是 '4bit' 或 '8bit'，收到 {mode}")

    logger.info("用 bitsandbytes %s 加载模型: %s", mode, model_path)
    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map="auto",
    )

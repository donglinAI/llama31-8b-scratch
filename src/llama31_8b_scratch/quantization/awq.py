"""AWQ：激活感知权重量化（惰性封装 autoawq）。

AWQ 是什么
==========
AWQ（Lin et al., 2023, *AWQ: Activation-aware Weight Quantization for
LLM Compression and Acceleration*）观察到两个事实：

1. **权重量化误差大的不是"权重大"的通道，而是"激活大"的通道**——因为
   输出 = Σ w_j · x_j，权重乘以一个它几乎不乘的小激活，对输出几乎没影响；
   反过来，即使权重小，只要对应激活经常很大，量化误差就会被放大。
2. **激活分布是稀疏的**：少数通道占了绝大部分激活幅度。

朴素 per-channel int4 量化不看激活，所以把"关键通道"也压成同样粗的网格，
误差就大。AWQ 的解法是：**量化前先把权重做一次可逆缩放，让关键通道在
量化网格上占更大的动态范围**。

数学（无跳步）
==============
对线性层 :math:`y = W x`，按输入通道 :math:`j` 选一个可学习的缩放系数
:math:`s_j`（标量，逐通道），定义：

.. math::
    w'_j = s_j \\cdot w_j, \\qquad x'_j = x_j / s_j

数学上 :math:`w'_j x'_j = w_j x_j` 完全不变（可逆）。我们接着对 :math:`W'`
做 int4 量化，推理时用 :math:`x'` 代入：

.. math:: y = W'_q \\, x' = \\Big(s_j\\, W_j\\Big)_q \\cdot \\frac{x_j}{s_j}

关键是怎么选 :math:`s_j`。AWQ 用校准集统计每个输入通道的平均激活幅度
:math:`\\bar a_j = \\mathrm{mean}(|x_j|)` 与平均权重幅度
:math:`\\bar w_j = \\mathrm{mean}(|W[:,j]|)`，然后：

.. math:: s_j = \\frac{\\bar a_j^{\\alpha}}{\\bar w_j^{1-\\alpha}}, \\quad
         \\alpha \\approx 0.5

直觉：激活大而权重小的通道（:math:`\\bar a_j` 大、:math:`\\bar w_j` 小），
:math:`s_j` 会变大，把权重整体放大到量化网格的精细区里；反之权重本来就
大、激活小的通道，:math:`s_j\\approx 1`，不浪费网格。

注意：AWQ **不真的修改推理时的激活**（那需要改算子），它只是把 :math:`s_j`
揉进量化 scale 里，让推理路径和普通 int4 一样。

autoawq 调用（惰性）
====================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["awq_quantize"]


def awq_quantize(
    model_path: str | Path,
    output_dir: str | Path,
    calibration_texts: Sequence[str],
    *,
    bits: int = 4,
    group_size: int = 128,
    version: str = "GEMM",
) -> str:
    """用 autoawq 对 HF 模型做 AWQ 量化并落盘。

    Args:
        model_path: 本地 HF 模型目录。
        output_dir: 输出目录。
        calibration_texts: 校准文本（autoawq 内部会用 tokenizer 切分）。
        bits: 通常为 4；8 一般不推荐（bnb 8bit 更成熟）。
        group_size: 常用 128。
        version: ``"GEMM"``（通用）或 ``"GEMV"``（老 kernel）。

    Returns:
        ``output_dir`` 字符串。
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "未检测到 autoawq。AWQ 量化请先安装：\n"
            "    pip install autoawq transformers accelerate\n"
        ) from exc

    model_path = str(model_path)
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }

    logger.info("加载待量化模型: %s", model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("开始 AWQ 校准与量化，样本数=%d", len(calibration_texts))
    model.quantize(
        list(calibration_texts),
        quant_config=quant_config,
    )

    logger.info("保存量化模型到: %s", output_dir)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

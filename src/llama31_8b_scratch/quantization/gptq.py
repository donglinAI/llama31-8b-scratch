"""GPTQ 二阶逐层权重量化（惰性封装 auto_gptq / optimum）。

GPTQ 是什么
===========
GPTQ（Frantar et al., 2023, *GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers*）把 "逐层、逐列" 的权重量化变成一个
**用二阶信息最小化输出误差**的算法。它是 OBQ（Optimal Brain Quantization）
在 Transformer 上的工程化版本。

直觉
----
朴素地直接把每个权重 :math:`w_{ij}` 量化成 int4，误差是孤立的；但同一层
里 8000 个权重的误差会**叠加**到输出上。GPTQ 的关键想法是：量化某一列
权重时产生的误差，**立刻用剩下还没量化的列去补偿**，让该层输出尽量不变。

数学（无跳步）
==============
设某线性层权重 :math:`W \\in \\mathbb R^{\\text{out}\\times\\text{in}}`，
该层输入激活 :math:`X \\in \\mathbb R^{n\\times\\text{in}}`（n 是校准 token 数）。

1. **Hessian**：该层输出对权重的二阶信息（去掉常数因子 2）：

   .. math:: H = X^\\top X \\in \\mathbb R^{\\text{in}\\times\\text{in}}

   它的元素 :math:`H_{ij}` 表示"第 i 个输入维度与第 j 个输入维度在输出上
   的共变"。

2. **按列量化 + 误差补偿**（GPTQ 按列 :math:`c = 1,\\dots,d` 推进）：

   - 量化第 c 列：:math:`q_c = \\mathrm{quant}(W[:,c])`，记残差
     :math:`\\delta_c = W[:,c] - q_c`；
   - 把残差按 H 的逆方向"摊"到后面还没量化的列上：

     .. math:: W[:, j] \\mathrel{+}= -\\frac{\\delta_c}{H_{cc}} H_{cj},
               \\quad j > c

   这样第 c 列量化带来的输出误差 :math:`\\delta_c \\cdot X[:,c]` 被后面
   列的调整抵消。

3. **数值上用 Cholesky**：直接求 :math:`H^{-1}` 不稳定；实际是先做
   :math:`H = L L^\\top`，再解三角系统，等价于用
   :math:`(H + \\epsilon I)^{-1}` 做上述更新（加正则是为了 H 半正定）。

为什么快
--------
OBQ 原本要对每个权重单独更新，复杂度 :math:`O(d^3)`；GPTQ 把它改成按列
一次性更新后面所有列，配合 Cholesky 是 :math:`O(d^3/6)` 量级，单张 A100
量化 175B 模型只要几小时。

group_size 是什么
-----------------
当 in_features 很大（如 11008）时，整列共用一个 scale 会让分布两端的权重
误差很大。GPTQ 把输入维切成每组 ``group_size``（常用 128）的块，每组独立
算 scale/zero-point，显著降低误差，代价是要多存几组 scale。

auto_gptq 调用（惰性，未装时给清晰提示）
========================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["gptq_quantize"]


def gptq_quantize(
    model_path: str | Path,
    output_dir: str | Path,
    calibration_texts: Sequence[str],
    *,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    dtype: str = "float16",
) -> str:
    """用 auto_gptq 对 HF 模型做 GPTQ 量化并落盘。

    Args:
        model_path: 本地 HF 模型目录（含 config.json + safetensors）。
        output_dir: 量化后保存目录。
        calibration_texts: 校准文本列表（几百段、每段几百 token 即可）。
            GPTQ 用它统计每一层的输入激活 H。
        bits: 4 或 8。
        group_size: 常用 128；0 表示按整张权重张量（per-channel）。
        desc_act: 是否按激活幅度降序逐列量化（True 略准，False 略快）。
        dtype: 校准期间模型权重 dtype。

    Returns:
        ``output_dir`` 字符串。

    Note:
        本函数**惰性 import** ``auto_gptq``。未安装时抛 ImportError 并提示
        ``pip install auto-gptq optimum``。不在文档里预填任何 perplexity /
        benchmark 数字——那些必须真机跑完才有意义。
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "未检测到 auto_gptq / transformers。GPTQ 量化请先安装：\n"
            "    pip install auto-gptq optimum transformers accelerate\n"
            "注意 auto_gptq 对 torch/CUDA 版本有要求，建议用官方指定组合。"
        ) from exc

    model_path = str(model_path)
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )
    logger.info("加载待量化模型: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )

    # 把校准文本转成 token 序列（auto_gptq 接受 list[dict("input_ids": ...)]）
    examples = []
    for text in calibration_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        examples.append({"input_ids": enc["input_ids"].squeeze(0)})

    logger.info("开始 GPTQ 校准与量化，样本数=%d", len(examples))
    model.quantize(examples, batch_size=1)

    logger.info("保存量化模型到: %s", output_dir)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir

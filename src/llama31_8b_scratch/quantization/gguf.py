"""GGUF：llama.cpp 模型格式与量化类型命令封装。

GGUF 是什么
===========
GGUF 是 `llama.cpp <https://github.com/ggml-org/llama.cpp>`_ 定义的模型文件
格式（GGML 的下一代）。它把权重、tokenizer 词表、超参数整张塞进一个二进制
文件，CPU / 移动端 / Apple Silicon 上跑 LLM 几乎都走这条路线。

和 GPTQ/AWQ/bnb 的区别
----------------------
- GPTQ/AWQ/bnb 是 **CUDA 生态**的量化，权重存成 int4/int8 张量，由 GPU kernel
  反量化；
- GGUF 是 **CPU/边缘生态**的量化，量化类型（Q4_K_M 等）是 llama.cpp 自己
  定义的块结构，由 ggml 算子在 CPU 上直接算。

两者不通用——你不能把 GPTQ 的 safetensors 直接喂给 llama.cpp，也不能把
GGUF 直接喂给 vLLM（vLLM 走自己的 GPTQ/AWQ 路径）。

常用量化类型（llama-quantize 子命令）
=====================================
+-----------+-----------+--------------------------------------+
| 类型      | 比特/权重 | 备注                                 |
+===========+===========+======================================+
| ``F16``   | 16        | 不量化，转换基线                     |
+-----------+-----------+--------------------------------------+
| ``Q8_0``  | 8.5       | 接近无损，CPU 推荐                   |
+-----------+-----------+--------------------------------------+
| ``Q4_K_M``| 4.8       | **体积/质量甜点**，社区默认推荐      |
+-----------+-----------+--------------------------------------+
| ``Q5_K_M``| 5.7       | 比 Q4 略准，体积略大                 |
+-----------+-----------+--------------------------------------+
| ``Q6_K``  | 6.6       | 高质量、可接受的体积                 |
+-----------+-----------+--------------------------------------+
| ``Q4_0``  | 4.5       | 老格式，不推荐；用 Q4_K_M 代替       |
+-----------+-----------+--------------------------------------+

为什么 K 系列比老的 Q4_0/Q4_1 好
--------------------------------
K-quants 把权重张量切成更小的块，块内用不同 scale（block-wise），并对
outlier 通道单独保留更高精度（这就是 "_K_" 的含义：kind / mixed precision
inside a block）。Q4_K_M 是 K 系列的中等档（Medium），是社区实测下来
体积/困惑度最平衡的一档。

转换两步走（llama.cpp 仓库自带脚本，本项目不重新实现）
=====================================================
1. **HF 模型 → F16 GGUF**：用 llama.cpp 仓库根目录的
   ``convert_hf_to_gguf.py``（老版本叫 ``convert.py``）：

   .. code-block:: bash

       python convert_hf_to_gguf.py ./hf-model-dir \\
           --outtype f16 \\
           --outfile model-f16.gguf

2. **F16 GGUF → 量化 GGUF**：用编译产物 ``llama-quantize``：

   .. code-block:: bash

       ./build/bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

本模块只负责**拼命令、做参数校验、可选地用 subprocess 调起**；
不假设 llama.cpp 装在某个固定路径，由调用方通过 ``llama_cpp_dir`` 指定。
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger(__name__)

# llama.cpp 支持的量化类型白名单（用于参数校验；未列出的类型命令也会拒绝）
GGUF_QUANT_TYPES = {
    "F32", "F16",
    "Q8_0", "Q8_1",
    "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS",
    "IQ2_S", "IQ2_M", "IQ4_NL", "IQ4_XS",
}

__all__ = [
    "GGUF_QUANT_TYPES",
    "GGUFCommands",
    "build_convert_command",
    "build_quantize_command",
    "run_gguf_pipeline",
]


@dataclass
class GGUFCommands:
    """拼装好的两条命令（list 形式，可直接给 subprocess 用）。"""

    convert: List[str]
    quantize: List[str]


def build_convert_command(
    hf_model_dir: str | Path,
    output_gguf: str | Path,
    *,
    llama_cpp_dir: str | Path,
    outtype: str = "f16",
    python_bin: str = "python3",
) -> List[str]:
    """构造 ``convert_hf_to_gguf.py`` 命令。

    Args:
        hf_model_dir: HF 格式模型目录（含 config.json + safetensors）。
        output_gguf: 输出 F16 GGUF 路径。
        llama_cpp_dir: llama.cpp 仓库根目录（含 ``convert_hf_to_gguf.py``）。
        outtype: ``f16`` / ``f32`` / ``bf16`` / ``q8_0`` 等。
        python_bin: Python 解释器。
    """
    script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not script.is_file():
        # 兼容老版本仓库：convert.py
        legacy = Path(llama_cpp_dir) / "convert.py"
        if legacy.is_file():
            script = legacy
        else:
            raise FileNotFoundError(
                f"在 {llama_cpp_dir} 下找不到 convert_hf_to_gguf.py 或 convert.py，"
                "请先 clone llama.cpp 仓库并指向其根目录。"
            )
    return [
        python_bin,
        str(script),
        str(hf_model_dir),
        "--outtype", outtype,
        "--outfile", str(output_gguf),
    ]


def build_quantize_command(
    input_gguf: str | Path,
    output_gguf: str | Path,
    quant_type: str = "Q4_K_M",
    *,
    llama_cpp_dir: str | Path,
) -> List[str]:
    """构造 ``llama-quantize`` 命令。

    Args:
        input_gguf: 上一步产出的 F16 GGUF。
        output_gguf: 输出量化 GGUF。
        quant_type: 必须在 :data:`GGUF_QUANT_TYPES` 白名单内。
        llama_cpp_dir: llama.cpp 仓库根目录（编译产物通常在 ``build/bin/``）。
    """
    if quant_type not in GGUF_QUANT_TYPES:
        raise ValueError(
            f"不认识的量化类型 {quant_type!r}；常见选项："
            f"Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16"
        )
    binary = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    if not binary.is_file():
        # 兼容不同构建目录布局
        for candidate in (
            Path(llama_cpp_dir) / "llama-quantize",
            Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize",
        ):
            if candidate.is_file():
                binary = candidate
                break
        else:
            raise FileNotFoundError(
                f"在 {llama_cpp_dir} 下找不到 llama-quantize，"
                "请先在 llama.cpp 仓库里 cmake 编译。"
            )
    return [str(binary), str(input_gguf), str(output_gguf), quant_type]


def run_gguf_pipeline(
    hf_model_dir: str | Path,
    output_dir: str | Path,
    *,
    llama_cpp_dir: str | Path,
    quant_type: str = "Q4_K_M",
    run: bool = False,
) -> GGUFCommands:
    """拼装（并可选执行）HF→F16→量化 GGUF 的完整命令。

    Args:
        hf_model_dir: HF 模型目录。
        output_dir: 中间 F16 GGUF 与最终量化 GGUF 都放在这里。
        llama_cpp_dir: llama.cpp 仓库根目录。
        quant_type: 量化类型，默认 ``Q4_K_M``。
        run: 是否真的用 subprocess 执行；默认 False，只返回命令让用户核对。

    Returns:
        :class:`GGUFCommands`，包含 convert 与 quantize 两条命令。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(hf_model_dir).name or "model"
    f16_path = output_dir / f"{name}-f16.gguf"
    q_path = output_dir / f"{name}-{quant_type}.gguf"

    convert_cmd = build_convert_command(
        hf_model_dir, f16_path, llama_cpp_dir=llama_cpp_dir
    )
    quant_cmd = build_quantize_command(
        f16_path, q_path, quant_type=quant_type, llama_cpp_dir=llama_cpp_dir
    )

    logger.info("转换命令: %s", shlex.join(convert_cmd))
    logger.info("量化命令: %s", shlex.join(quant_cmd))

    if run:
        logger.info("执行 HF -> F16 GGUF ...")
        subprocess.run(convert_cmd, check=True)
        logger.info("执行 %s 量化 ...", quant_type)
        subprocess.run(quant_cmd, check=True)
        logger.info("完成: %s", q_path)

    return GGUFCommands(convert=convert_cmd, quantize=quant_cmd)

"""量化子包：手写量化数学 + GPTQ/AWQ/bnb/GGUF 工业封装。

分两层：

1. **手写教学层**（纯 numpy，CPU 可跑、数值可验证）：
   - :mod:`llama31_8b_scratch.quantization.core`  对称/仿射量化、scale/zero-point、
     int8/int4、quant→dequant 与误差边界
2. **工业封装层**（全部惰性导入，缺库时给清晰提示）：
   - :mod:`llama31_8b_scratch.quantization.gptq`     GPTQ 二阶逐层量化（auto_gptq/optimum）
   - :mod:`llama31_8b_scratch.quantization.awq`     AWQ 激活感知缩放（autoawq）
   - :mod:`llama31_8b_scratch.quantization.bnb`     bitsandbytes 8bit/4bit NF4
   - :mod:`llama31_8b_scratch.quantization.gguf`    llama.cpp GGUF 转换与 Q4_K_M 等命令

顶层不 import 任何可选重依赖。
"""

from __future__ import annotations

from llama31_8b_scratch.quantization import (  # noqa: F401
    awq,
    bnb,
    core,
    gguf,
    gptq,
)

__all__ = ["core", "gptq", "awq", "bnb", "gguf"]

"""手写对称 / 仿射均匀量化的数学核心（纯 numpy，CPU 可验证）。

为什么先手写
============
GPTQ / AWQ / bitsandbytes / GGUF 这些工业库都在做同一件事的不同工程优化：
把 fp16/bf16 的权重或激活压成 int8/int4，再反量化回去参与计算。要真正读懂
这些库在干什么、误差从哪来、为什么 group size 很重要，必须先把最朴素的
**均匀量化 + 反量化**自己写一遍。

数学记号（无跳步）
==================
设浮点张量为 :math:`x \\in \\mathbb R^n`，我们要把它映射到 ``b`` 位有符号
整数网格上，整数范围为：

.. math::
    q_{\\min} = -2^{b-1}, \\quad q_{\\max} = 2^{b-1} - 1

例如 b=8 时 ``[-128, 127]``，b=4 时 ``[-8, 7]``。

**1. 仿射（非对称）量化**：把浮点区间 ``[x_min, x_max]`` 线性映射到整数网格。

.. math::
    s &= \\frac{x_{\\max} - x_{\\min}}{q_{\\max} - q_{\\min}}
       \\quad\\text{(scale，浮点)} \\\\
    z &= \\operatorname{round}\\!\\left(q_{\\min} - \\frac{x_{\\min}}{s}\\right)
       \\quad\\text{(zero-point，整数)} \\\\
    q &= \\operatorname{clip}\\!\\left(
          \\operatorname{round}\\!\\left(\\frac{x}{s}\\right) + z,
          q_{\\min}, q_{\\max}\\right)

反量化（dequant）：

.. math:: \\hat x = s \\cdot (q - z)

**2. 对称量化**：强制 zero-point = 0，区间关于 0 对称，适合权值（权值常以 0
为中心）：

.. math::
    m &= \\max(|x_{\\min}|, |x_{\\max}|) \\\\
    s &= \\frac{m}{q_{\\max}} \\\\
    q &= \\operatorname{clip}\\!\\left(\\operatorname{round}\\!\\left(\\frac{x}{s}\\right),
          q_{\\min}, q_{\\max}\\right), \\quad z = 0 \\\\
    \\hat x &= s \\cdot q

误差边界（必须记住）
====================
均匀量化的最大逐元素绝对误差不超过半个网格步长：

.. math::
    |\\hat x - x| \\le \\frac{s}{2}

这是因为 ``round`` 引入的误差最大为 0.5 个整数单位，再乘回 scale 就是 s/2。
**int4 相对 int8 的 scale 大约大一倍，误差大约也大一倍**——这就是为什么
4bit 量化需要 group size、激活感知缩放（AWQ）、二阶误差补偿（GPTQ）这些
技术来把误差压回去。

实现约定
========
- 本模块只做 per-tensor 量化（整个张量一个 scale/zero-point）。
  工业上常见的 per-channel / per-group 只是把这个函数在最后一维分块调用。
- 不做 bit-packing（int4 仍以 int8/int16 存储）。bit-packing 是工程细节，
  不影响数学，文档里单独讲。
- 输入用 ``np.ndarray``；torch 调用方先 ``tensor.detach().cpu().numpy()``。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "QuantResult",
    "int_range",
    "affine_quantize",
    "affine_dequantize",
    "symmetric_quantize",
    "symmetric_dequantize",
    "quantization_error",
]


@dataclass
class QuantResult:
    """量化结果：整数张量 + scale + zero-point。"""

    q: np.ndarray            # 整数张量（dtype=int64，未做 bit-pack）
    scale: float
    zero_point: int
    num_bits: int

    def dequantize(self) -> np.ndarray:
        """反量化回浮点。"""
        return self.scale * (self.q - self.zero_point)


def int_range(num_bits: int, signed: bool = True) -> Tuple[int, int]:
    """返回 ``(q_min, q_max)``。

    - signed=True（默认）：``[-2^(b-1), 2^(b-1)-1]``
    - signed=False：``[0, 2^b - 1]``（常用在权重对称量化的另一种写法）
    """
    if num_bits < 2 or num_bits > 16:
        raise ValueError(f"num_bits 应在 [2,16]，收到 {num_bits}")
    if signed:
        q_min = -(2 ** (num_bits - 1))
        q_max = 2 ** (num_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2 ** num_bits - 1
    return q_min, q_max


def affine_quantize(
    x: np.ndarray, num_bits: int = 8, signed: bool = True
) -> QuantResult:
    """仿射（非对称）均匀量化。"""
    x = np.asarray(x, dtype=np.float64)
    q_min, q_max = int_range(num_bits, signed=signed)

    x_min = float(x.min())
    x_max = float(x.max())
    if x_max == x_min:
        # 常数张量：选 scale=|c|，q=sign(c)，zero_point=0，可精确还原
        c = x_min
        scale = abs(c) if c != 0.0 else 1.0
        q_val = 0 if c == 0.0 else (1 if c > 0 else -1)
        q = np.full_like(x, q_val, dtype=np.int64)
        return QuantResult(q=q, scale=float(scale), zero_point=0,
                           num_bits=num_bits)

    scale = (x_max - x_min) / (q_max - q_min)
    # z = round(q_min - x_min / scale)，把 x_min 对齐到 q_min
    zero_point = int(round(q_min - x_min / scale))
    # clip zero_point 到合法范围（极端分布下可能越界）
    zero_point = max(q_min, min(q_max, zero_point))

    q = np.round(x / scale) + zero_point
    q = np.clip(q, q_min, q_max).astype(np.int64)
    return QuantResult(q=q, scale=float(scale), zero_point=zero_point,
                       num_bits=num_bits)


def affine_dequantize(r: QuantResult) -> np.ndarray:
    """仿射反量化（等价于 ``r.dequantize()``，保留为独立函数便于教学）。"""
    return r.scale * (r.q - r.zero_point)


def symmetric_quantize(
    x: np.ndarray, num_bits: int = 8
) -> QuantResult:
    """对称量化：zero_point 强制为 0，区间关于 0 对称。"""
    x = np.asarray(x, dtype=np.float64)
    q_min, q_max = int_range(num_bits, signed=True)

    max_abs = float(max(abs(x.min()), abs(x.max())))
    if max_abs == 0.0:
        return QuantResult(q=np.zeros_like(x, dtype=np.int64), scale=1.0,
                           zero_point=0, num_bits=num_bits)

    scale = max_abs / q_max
    q = np.round(x / scale)
    q = np.clip(q, q_min, q_max).astype(np.int64)
    return QuantResult(q=q, scale=float(scale), zero_point=0,
                       num_bits=num_bits)


def symmetric_dequantize(r: QuantResult) -> np.ndarray:
    """对称反量化：``x_hat = scale * q``。"""
    return r.scale * r.q


def quantization_error(
    x: np.ndarray, x_hat: np.ndarray
) -> dict:
    """量化误差统计：max_abs、mean_abs、relative_l2。"""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    diff = x_hat - x
    denom = float(np.linalg.norm(x)) + 1e-12
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / denom),
    }

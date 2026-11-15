"""评测指标聚合核心：均值、总体/样本标准差、准确率、置信区间。

本模块**只依赖标准库与 numpy**，不 import torch，也不依赖 scipy / evaluate / sklearn，
便于在任意环境下被单测、被其它模块复用。所有函数都接受 ``array-like``（list / tuple /
numpy 一维数组），返回标量或 (low, high) 元组。

数学记号（设样本为 :math:`x_1,\\dots,x_n`，均值 :math:`\\bar x`）：

- 均值：

  .. math:: \\bar x = \\frac{1}{n}\\sum_{i=1}^{n} x_i

- 样本标准差（无偏，分母 :math:`n-1`）：

  .. math:: s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(x_i-\\bar x)^2}

- 95% 置信区间（正态近似）：

  .. math:: \\bar x \\pm z_{1-\\alpha/2}\\cdot \\frac{s}{\\sqrt{n}}

  其中 :math:`z_{0.975}\\approx 1.95996`。本模块用逆误差函数反解任意
  :math:`\\alpha` 对应的分位数，因此 90%/95%/99% 都能算，不需要硬编码查表。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np

# array-like 的宽松类型标注
Number = Union[int, float, np.number]
ScalarSequence = Union[Sequence[Number], np.ndarray]

__all__ = ["mean", "std", "accuracy", "confidence_interval", "normal_ppf"]


def mean(values: ScalarSequence) -> float:
    """返回 ``values`` 的算术平均；空序列抛 ``ValueError``。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mean() 接收到空序列")
    return float(arr.mean())


def std(values: ScalarSequence, ddof: int = 1) -> float:
    """返回样本标准差（默认 ``ddof=1`` 无偏估计；总体标准差传 ``ddof=0``）。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= ddof:
        raise ValueError(
            f"std() 需要 length > ddof，当前 length={arr.size}, ddof={ddof}"
        )
    return float(arr.std(ddof=ddof))


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """分类准确率 = 预测与标签相等的比例。

    Args:
        preds: 预测类别索引序列。
        labels: 真值类别索引序列，长度必须与 ``preds`` 一致。

    Returns:
        ``[0, 1]`` 之间的浮点数；空序列抛 ``ValueError``。
    """
    p = np.asarray(preds)
    y = np.asarray(labels)
    if p.shape != y.shape:
        raise ValueError(f"preds/labels 形状不一致: {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("accuracy() 接收到空序列")
    return float((p == y).mean())


# ---------------------------------------------------------------------------
# 正态分布分位数（逆误差函数反解），仅用标准库 math，避免依赖 scipy
# ---------------------------------------------------------------------------
def _erf(x: float) -> float:
    """标准库的 erf 包装（Python 3.x 自带）。"""
    return math.erf(x)


def _erfinv(y: float) -> float:
    """逆误差函数 erfinv(y)，|y|<1。用 Newton 迭代，初值用有理逼近。

    这里不追求极致精度，6~8 次迭代即可到 1e-9，足够置信区间使用。
    """
    if not -1.0 < y < 1.0:
        raise ValueError(f"erfinv 定义域为 (-1, 1)，收到 {y}")
    if y == 0.0:
        return 0.0

    # 初值：Winit 来自 A&S 7.1.26（Newman 逼近）
    a = 0.147
    ln = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    t2 = ln / a
    sign = 1.0 if y >= 0.0 else -1.0
    x = sign * math.sqrt(math.sqrt(t1 * t1 - t2) - t1)

    # Newton: 解 f(x)=erf(x)-y=0, f'(x)=2/sqrt(pi)*exp(-x^2)
    for _ in range(8):
        fx = _erf(x) - y
        dx = fx / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
        x -= dx
        if abs(dx) < 1e-12:
            break
    return x


def normal_ppf(p: float) -> float:
    """标准正态分布的分位数函数 :math:`\\Phi^{-1}(p)`。

    利用 :math:`\\Phi^{-1}(p)=\\sqrt{2}\\,\\mathrm{erfinv}(2p-1)`。
    例如 ``normal_ppf(0.975) ≈ 1.959964``。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf 要求 0<p<1，收到 {p}")
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def confidence_interval(
    values: ScalarSequence, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """均值的正态近似置信区间。

    Args:
        values: 样本标量序列。
        confidence: 置信水平，例如 0.90 / 0.95 / 0.99。

    Returns:
        ``(mean, lower, upper)`` 三元组。样本量 ``n`` 很小时正态近似
        并不严谨，但在大模型评测几百到几千样本的场景下是 lm-eval、
        OpenLLM 报告的通行做法。
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        raise ValueError(f"confidence_interval 至少需要 2 个样本，当前 n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 必须在 (0,1)，收到 {confidence}")

    mu = float(arr.mean())
    s = float(arr.std(ddof=1))
    z = normal_ppf(0.5 + confidence / 2.0)  # 双侧分位数
    half = z * s / math.sqrt(n)
    return mu, mu - half, mu + half

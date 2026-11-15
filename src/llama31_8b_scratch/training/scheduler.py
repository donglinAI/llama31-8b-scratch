r"""手写学习率调度器：Warmup + Cosine Decay（闭式公式）。

为什么手写而不直接用 ``torch.optim.lr_scheduler``：
- 教学上要把"每个 step 的学习率是多少"用一行闭式公式讲清楚，便于在 paper / 面试里推导；
- 工业上 HF / DeepSpeed 的 cosine 实现本质就是同一个公式，手写后再去读它们的源码会非常快；
- 与梯度累积解耦：调度器只面对"已完成的优化器更新次数"（global optimizer step），
  而不是 forward 次数，这样无论梯度累积多大、卡数多少，曲线形状都一致。

闭式公式（与 ``num_cycles=1`` 的标准 cosine 一致）：

设：
- ``T`` = 总优化步数（global step 总数，已含梯度累积与多卡）；
- ``W`` = warmup 步数；
- ``eta_min`` = 最小学习率占峰值的比例（``min_lr_ratio``）；
- ``base_lr`` = 峰值学习率（也就是优化器初始化时的 ``lr``）。

对第 ``t`` 个已完成优化步（``t = 0, 1, ..., T``）：

1. warmup 阶段（``t < W``）：

.. math::
    \\eta(t) = base\_lr \\cdot \\frac{t}{W}

   线性从 0 爬到 ``base_lr``；``t=0`` 时为 0，避免初始大步长把权重打飞。

2. cosine 衰减阶段（``t >= W``）：令进度

.. math::
    p = \\mathrm{clip}\\!\\left(\\frac{t - W}{T - W},\\ 0,\\ 1\\right)

   则

.. math::
    \\eta(t) = base\_lr \\cdot \\Big[\\eta_{min} + (1 - \\eta_{min})\\
             \\cdot \\tfrac{1}{2}\\big(1 + \\cos(\\pi \\cdot p)\\big)\\Big]

   验证端点：
   - ``p=0``（刚结束 warmup）：``cos(0)=1``，因子 = 1，``eta = base_lr``，与 warmup 末端连续；
   - ``p=1``（训练结束）：``cos(pi)=-1``，因子 = 0，``eta = base_lr * eta_min``。

这正是 HF ``get_cosine_schedule_with_warmup`` 的数学内核，区别仅在于 HF 用
``LambdaLR`` 把这个公式包成 ``lambda``，我们这里直接写成显式函数，便于单测在
warmup / 稳态 / 末端三个点做数值闭式校验。
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch.optim import Optimizer

__all__ = ["CosineWarmupScheduler", "build_scheduler"]


class CosineWarmupScheduler:
    """Warmup + Cosine 学习率调度器（手写、闭式、可单测）。

    用法::

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = CosineWarmupScheduler(
            optimizer, base_lr=3e-4, warmup_steps=100, total_steps=10000,
            min_lr_ratio=0.1,
        )
        for batch in loader:
            loss.backward()
            optimizer.step()
            sched.step()   # 每个优化器 step 后调用一次（不是每个 micro-batch）

    注意：``step()`` 必须在 ``optimizer.step()`` 之后调用，且**每个优化器更新只调一次**。
    当梯度累积开启时，一次优化器更新对应 ``GAS`` 次 forward，``sched.step()`` 只在
    ``optimizer.step()`` 那次调用，从而让 schedule 的单位是"优化步"而不是"forward 步"。

    Args:
        optimizer: 已构造好的优化器，各 param_group 的 ``lr`` 会被就地改写。
        base_lr: 峰值学习率；为 None 时取 optimizer 第 0 个 param_group 的初始 lr。
        warmup_steps: 线性 warmup 的优化步数 ``W``。
        total_steps: 训练总优化步数 ``T``。
        min_lr_ratio: 末端最小学习率占峰值比例 ``eta_min``，默认 0.1（即末端为 10% 峰值）。
        last_step: 断点续训时恢复到的步数；构造后内部计数器对齐到该值。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float | Sequence[float] | None = None,
        warmup_steps: int = 0,
        total_steps: int = 1,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0:
            raise ValueError("warmup_steps>=0 且 total_steps>0")
        if not 0.0 <= min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio 必须在 [0, 1) 之间")

        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)

        # 每个 param_group 允许不同 base_lr（例如 LoRA 用更大 lr、embedding 用更小 lr）
        if base_lr is None:
            self.base_lrs: List[float] = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
        elif isinstance(base_lr, (int, float)):
            self.base_lrs = [float(base_lr)] * len(optimizer.param_groups)
        else:
            self.base_lrs = [float(x) for x in base_lr]
        if len(self.base_lrs) != len(optimizer.param_groups):
            raise ValueError(
                f"base_lr 个数({len(self.base_lrs)}) 与 param_group 数"
                f"({len(optimizer.param_groups)}) 不一致"
            )

        # 内部计数器：当前已经"完成"的优化步数
        self.last_step = last_step
        # 构造时先对齐到 last_step（断点续训），再在 step() 里自增
        self._apply()

    # ------------------------------------------------------------------
    # 核心闭式公式
    # ------------------------------------------------------------------
    def lr_at(self, step: int, base_lr: float) -> float:
        """返回第 ``step`` 个已完成优化步对应的学习率（闭式、可独立单测）。

        Args:
            step: 已完成的优化步数 ``t``，``0 <= step <= total_steps``。
            base_lr: 该 param_group 的峰值学习率。
        """
        w = self.warmup_steps
        t_total = self.total_steps

        # 1) warmup：线性 ramp
        if w > 0 and step < w:
            return base_lr * (step / w)

        # 2) cosine 衰减
        if t_total <= w:
            # 退化情形：total <= warmup，直接保持峰值
            return base_lr

        # p = (t - W) / (T - W)，裁剪到 [0, 1]，防止最后多跑几步越界
        progress = (step - w) / (t_total - w)
        progress = min(1.0, max(0.0, progress))

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor)

    # ------------------------------------------------------------------
    # Optimizer 接口
    # ------------------------------------------------------------------
    def _apply(self) -> None:
        """把当前 ``last_step`` 对应的学习率写回各 param_group。"""
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.lr_at(self.last_step, base_lr)

    def step(self) -> None:
        """在 ``optimizer.step()`` 之后调用一次，推进一个优化步。"""
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> List[float]:
        """返回当前各 param_group 的学习率（监控 / 日志用）。"""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """断点保存：只需要步数与 base_lrs，不依赖 optimizer 内部状态。"""
        return {
            "last_step": self.last_step,
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state: dict) -> None:
        """断点恢复：对齐步数后重新写一次 lr。"""
        self.last_step = int(state["last_step"])
        self._apply()


def build_scheduler(
    optimizer: Optimizer,
    base_lr: float | Sequence[float],
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    last_step: int = -1,
) -> CosineWarmupScheduler:
    """工厂函数：从配置字段构造调度器。"""
    return CosineWarmupScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
        last_step=last_step,
    )

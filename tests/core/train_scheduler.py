"""CosineWarmupScheduler 的数值闭式校验（CPU、纯 torch）。

校验点：
1. warmup 阶段：lr 线性从 0 爬到 peak；
2. warmup 结束那一步：lr == peak（连续）；
3. 训练中段（p=0.5）：cos(pi/2)=0，lr == peak * min_lr_ratio + (1-min)*peak*0.5
   = peak * (0.5 + 0.5*min_lr_ratio)；
4. 训练最后一步（t=total）：lr == peak * min_lr_ratio；
5. 超过 total 后：被裁剪，不再下降。
"""

from __future__ import annotations

import math
import unittest

import torch

from llama31_8b_scratch.training.scheduler import CosineWarmupScheduler


def _make_opt(lr: float = 1.0):
    p = torch.nn.Parameter(torch.zeros(2, 3))
    return torch.optim.SGD([p], lr=lr)


class TestCosineWarmupScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.peak = 1.0
        self.warmup = 10
        self.total = 100
        self.min_ratio = 0.1
        self.sched = CosineWarmupScheduler(
            _make_opt(self.peak),
            base_lr=self.peak,
            warmup_steps=self.warmup,
            total_steps=self.total,
            min_lr_ratio=self.min_ratio,
            last_step=-1,
        )

    def test_warmup_linear_ramp(self):
        # 构造时 last_step=-1；step() 先自增再应用。
        # 第 1 次 step 后 last_step=0，第 5 次 step 后 last_step=4。
        for _ in range(5):
            self.sched.step()
        # t=4：lr = peak * 4/10 = 0.4
        self.assertAlmostEqual(self.sched.lr_at(4, self.peak), 0.4, places=6)
        self.assertAlmostEqual(self.sched.get_last_lr()[0], 0.4, places=6)
        # 闭式公式本身在 t=5 时应为 0.5（不依赖 step 计数）
        self.assertAlmostEqual(self.sched.lr_at(5, self.peak), 0.5, places=6)

    def test_peak_at_warmup_end(self):
        # t = warmup = 10：lr 应该 == peak（cos(0)=1）
        for _ in range(11):
            self.sched.step()  # -> last_step = 10
        self.assertAlmostEqual(self.sched.lr_at(10, self.peak), self.peak, places=6)

    def test_midpoint_cosine(self):
        # t = 55（warmup 之后过半）：p = (55-10)/(100-10) = 0.5
        # cos(pi/2) = 0 -> lr = peak*(min_ratio + (1-min)*0.5)
        expected = self.peak * (self.min_ratio + (1 - self.min_ratio) * 0.5)
        self.assertAlmostEqual(
            self.sched.lr_at(55, self.peak), expected, places=6
        )

    def test_endpoint_min_lr(self):
        # t = total = 100：p=1, cos(pi)=-1 -> lr = peak*min_ratio
        expected = self.peak * self.min_ratio
        self.assertAlmostEqual(
            self.sched.lr_at(self.total, self.peak), expected, places=6
        )

    def test_clamped_after_total(self):
        # t > total：应被裁剪，保持 min_lr
        expected = self.peak * self.min_ratio
        self.assertAlmostEqual(
            self.sched.lr_at(self.total + 50, self.peak), expected, places=6
        )

    def test_no_warmup_starts_at_peak(self):
        sched = CosineWarmupScheduler(
            _make_opt(0.5), base_lr=0.5, warmup_steps=0, total_steps=100,
            min_lr_ratio=0.0, last_step=-1,
        )
        # t=0 且 warmup=0：直接进 cosine，p=0 -> lr=0.5
        sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.5, places=6)

    def test_state_dict_roundtrip(self):
        for _ in range(20):
            self.sched.step()
        # 从 last_step=-1 起，20 次 step 后 last_step=19
        self.assertEqual(self.sched.last_step, 19)
        state = self.sched.state_dict()
        sched2 = CosineWarmupScheduler(
            _make_opt(self.peak), base_lr=self.peak, warmup_steps=self.warmup,
            total_steps=self.total, min_lr_ratio=self.min_ratio,
        )
        sched2.load_state_dict(state)
        self.assertEqual(sched2.last_step, 19)
        self.assertAlmostEqual(
            sched2.lr_at(19, self.peak), self.sched.lr_at(19, self.peak), places=6
        )


if __name__ == "__main__":
    unittest.main()

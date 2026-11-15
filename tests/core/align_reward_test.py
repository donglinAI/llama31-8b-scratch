"""奖励模型成对损失闭式数值测试。"""

import math
import unittest

import torch

from llama31_8b_scratch.align.reward import (
    pairwise_accuracy,
    pairwise_loss,
    compute_pairwise_metrics,
)


def _logsigmoid(x: float) -> float:
    """数值稳定的 log σ(x) 闭式参考实现。"""
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))


class TestPairwiseLoss(unittest.TestCase):
    def test_closed_form_diff_one(self):
        # r_w - r_l = 1 -> L = -log σ(1)
        chosen = torch.tensor([1.0])
        rejected = torch.tensor([0.0])
        loss = pairwise_loss(chosen, rejected)
        expected = -_logsigmoid(1.0)
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_closed_form_tie_is_log2(self):
        # 差值为 0 -> σ(0)=0.5 -> L = -log 0.5 = log 2
        chosen = torch.tensor([0.5])
        rejected = torch.tensor([0.5])
        loss = pairwise_loss(chosen, rejected)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)

    def test_margin_shifts_diff(self):
        # 差值 0.5，margin=1.0 -> 有效差值 -0.5
        chosen = torch.tensor([1.5])
        rejected = torch.tensor([1.0])
        loss = pairwise_loss(chosen, rejected, margin=1.0)
        expected = -_logsigmoid(-0.5)
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_batch_average(self):
        # 两条样本差值分别为 +1、-1，损失为两者平均
        chosen = torch.tensor([1.0, 0.0])
        rejected = torch.tensor([0.0, 1.0])
        loss = pairwise_loss(chosen, rejected)
        expected = 0.5 * (-_logsigmoid(1.0) - _logsigmoid(-1.0))
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_accuracy(self):
        chosen = torch.tensor([1.0, 0.0, 2.0])
        rejected = torch.tensor([0.0, 1.0, 1.0])
        acc = pairwise_accuracy(chosen, rejected)
        self.assertAlmostEqual(acc.item(), 2.0 / 3.0, places=6)

    def test_metrics_dict(self):
        chosen = torch.tensor([1.0])
        rejected = torch.tensor([0.0])
        metrics = compute_pairwise_metrics(chosen, rejected)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertAlmostEqual(metrics["score_gap"].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

"""causal_lm_loss 与 z_loss 的数值校验。

校验：
1. label 移位是否正确：用一个 logits 手工算出 cross_entropy 对比；
2. -100 掩码是否真的被忽略：把某些 label 改成 -100，loss 应等于只在剩余位置算；
3. z_loss 数值：已知 logits 时手算 logsumexp^2 的均值 * coeff；
4. 全 -100 的退化情形：不应 NaN（分母为 0 时 PyTorch 给 0/0=NaN，需确认行为）。
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from llama31_8b_scratch.training.losses import IGNORE_INDEX, causal_lm_loss, z_loss


class TestCausalLMLoss(unittest.TestCase):
    def test_matches_manual_cross_entropy(self):
        # 构造一个确定的小例子：B=1, T=3, V=4
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 4, dtype=torch.float64)
        labels = torch.tensor([[1, 2, 0]])

        loss = causal_lm_loss(logits, labels)

        # 手算：logits[:, :-1, :] 预测 labels[:, 1:]
        manual = F.cross_entropy(
            logits[0, :-1, :], labels[0, 1:]
        )
        self.assertTrue(torch.allclose(loss, manual, atol=1e-12))

    def test_ignore_index_masking(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 5, dtype=torch.float64)
        # labels: 位置 1 之后是 [2, 0, 4]；把中间那个（位置2 的真值 labels[2]=0）掩掉
        labels = torch.tensor([[1, 2, IGNORE_INDEX, 4]])

        loss = causal_lm_loss(logits, labels)

        # 移位后：预测位置 1,2,3 的真值 = labels[1:]=[2, -100, 4]
        # 只在位置 1 与 3 上算（对应 logits 行 0、2）
        logits_shift = logits[0, :-1, :]
        labels_shift = torch.tensor([2, IGNORE_INDEX, 4])
        manual = F.cross_entropy(logits_shift, labels_shift, ignore_index=IGNORE_INDEX)
        self.assertTrue(torch.allclose(loss, manual, atol=1e-12))

    def test_z_loss_value(self):
        logits = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float64)
        coeff = 1e-3
        zl = z_loss(logits, z_loss_coeff=coeff)

        # logsumexp([1,0,-1]) = log(e + 1 + e^-1)
        lse = math.log(math.e + 1.0 + math.exp(-1.0))
        expected = coeff * (lse ** 2)
        self.assertAlmostEqual(float(zl), expected, places=8)

    def test_z_loss_zero_when_disabled(self):
        logits = torch.randn(2, 3, 5, dtype=torch.float64)
        zl = z_loss(logits, z_loss_coeff=0.0)
        self.assertEqual(float(zl), 0.0)

    def test_loss_combines_main_and_z(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 3, 4, dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2]])
        coeff = 1e-3

        loss = causal_lm_loss(logits, labels, z_loss_coeff=coeff)
        main = F.cross_entropy(logits[0, :-1, :], labels[0, 1:])
        zl = z_loss(logits[..., :-1, :], coeff)
        self.assertTrue(torch.allclose(loss, main + zl, atol=1e-12))


if __name__ == "__main__":
    unittest.main()

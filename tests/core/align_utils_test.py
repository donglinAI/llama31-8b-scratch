"""对齐通用工具单元测试：gather_logprobs、掩码平均、优势归一化。

全部使用合成张量，CPU 运行，不依赖 model / data 代理。
"""

import math
import unittest

import torch

from llama31_8b_scratch.align.utils import (
    gather_logprobs,
    masked_mean,
    normalize_advantages,
    sequence_logprobs,
)


class TestGatherLogprobs(unittest.TestCase):
    def test_uniform_logits_gather_correct(self):
        # logits 全 0 -> 词表均匀分布，每个 token 对数概率为 -log V
        logits = torch.zeros(1, 2, 3)
        labels = torch.tensor([[0, -100]])
        token_logps, mask = gather_logprobs(logits, labels)

        self.assertTrue(mask.tolist() == [[True, False]])
        self.assertAlmostEqual(token_logps[0, 0].item(), -math.log(3.0), places=6)
        self.assertEqual(token_logps[0, 1].item(), 0.0)

    def test_gather_matches_log_softmax(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 5)
        labels = torch.tensor([[0, 1, 2, -100], [3, 4, -100, 1]])
        token_logps, mask = gather_logprobs(logits, labels)

        ref = torch.log_softmax(logits, dim=-1)
        for b in range(2):
            for t in range(4):
                if mask[b, t]:
                    self.assertAlmostEqual(
                        token_logps[b, t].item(),
                        ref[b, t, labels[b, t]].item(),
                        places=6,
                    )
                else:
                    self.assertEqual(token_logps[b, t].item(), 0.0)

    def test_sequence_logprobs_sum(self):
        token_logps = torch.tensor([[-0.5, -0.3, 0.0], [-0.2, 0.0, -0.1]])
        mask = torch.tensor([[True, True, False], [True, False, True]])
        seq_lp = sequence_logprobs(token_logps, mask)
        self.assertAlmostEqual(seq_lp[0].item(), -0.8, places=6)
        self.assertAlmostEqual(seq_lp[1].item(), -0.3, places=6)

    def test_shape_mismatch_raises(self):
        logits = torch.zeros(1, 3, 5)
        labels = torch.zeros(1, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            gather_logprobs(logits, labels)


class TestMaskedMean(unittest.TestCase):
    def test_masked_mean_ignores_padding(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        self.assertAlmostEqual(masked_mean(values, mask).item(), 1.5, places=6)

    def test_masked_mean_dim(self):
        values = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        col_mean = masked_mean(values, mask, dim=0)
        self.assertAlmostEqual(col_mean[0].item(), 1.5, places=6)
        self.assertAlmostEqual(col_mean[1].item(), 4.0, places=6)


class TestNormalizeAdvantages(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normed = normalize_advantages(adv)
        self.assertAlmostEqual(normed.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(normed.std(unbiased=False).item(), 1.0, places=6)

    def test_masked_normalize(self):
        adv = torch.tensor([1.0, 3.0, 100.0])
        mask = torch.tensor([True, True, False])
        normed = normalize_advantages(adv, mask=mask)
        self.assertAlmostEqual(normed[2].item(), 0.0)
        self.assertAlmostEqual(normed[:2].mean().item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

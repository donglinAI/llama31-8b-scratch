"""GRPO 组内相对优势与策略损失测试（DeepSeekMath 2024 / DeepSeek-R1 2025）。"""

import unittest

import torch

from llama31_8b_scratch.align.grpo import compute_grpo_advantages, grpo_loss


class TestGroupAdvantages(unittest.TestCase):
    def test_zero_mean_unit_std_per_group(self):
        # 两组，每组 3 条；第二组奖励全相同 -> 标准差为 0，优势应全 0
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
        adv = compute_grpo_advantages(rewards, group_size=3)

        g1 = adv[:3]
        g2 = adv[3:]
        self.assertAlmostEqual(g1.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(g1.std(unbiased=False).item(), 1.0, places=6)
        self.assertTrue(torch.allclose(g2, torch.zeros_like(g2), atol=1e-6))

    def test_advantage_values_closed_form(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        adv = compute_grpo_advantages(rewards, group_size=3)
        std = (2.0 / 3.0) ** 0.5
        self.assertAlmostEqual(adv[0].item(), -1.0 / std, places=6)
        self.assertAlmostEqual(adv[1].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[2].item(), 1.0 / std, places=6)

    def test_non_divisible_raises(self):
        with self.assertRaises(ValueError):
            compute_grpo_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=2)


class TestGRPOLoss(unittest.TestCase):
    def test_policy_loss_finite_no_value_head(self):
        torch.manual_seed(0)
        N, T = 4, 3
        new_logp = torch.randn(N, T)
        old_logp = torch.randn(N, T)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(N, T, dtype=torch.bool)

        out = grpo_loss(new_logp, old_logp, adv, mask, clip_ratio=0.2)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertAlmostEqual(out["kl"].item(), 0.0, places=6)
        self.assertNotIn("value_loss", out)

    def test_kl_penalty_with_ref(self):
        new_logp = torch.zeros(2, 2)
        old_logp = torch.zeros(2, 2)
        ref_logp = torch.zeros(2, 2)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 2, dtype=torch.bool)
        out = grpo_loss(
            new_logp, old_logp, adv, mask, ref_logps=ref_logp,
            kl_coef=0.1, kl_estimator="k3",
        )
        # 策略与参考一致 -> k3 KL = 0
        self.assertAlmostEqual(out["kl"].item(), 0.0, places=6)
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()

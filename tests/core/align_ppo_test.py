"""PPO 手写核心测试：GAE 手算小例、策略裁剪行为、价值裁剪、组合损失。"""

import unittest

import torch

from llama31_8b_scratch.align.ppo import (
    clipped_policy_loss,
    clipped_value_loss,
    compute_gae,
    ppo_loss,
)


class TestGAE(unittest.TestCase):
    def test_gae_terminal_bootstrap_zero(self):
        # B=1, T=2；γ=1, λ=1；d_1=1 终止，末尾不引导
        # δ_1 = r1 + 0 - V1 = 0；A_1 = 0
        # δ_0 = r0 + V1 - V0 = 1；A_0 = δ_0 + A_1 = 1
        rewards = torch.tensor([[1.0, 0.0]])
        values = torch.tensor([[0.0, 0.0]])
        dones = torch.tensor([[0.0, 1.0]])
        adv, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

        self.assertAlmostEqual(adv[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(adv[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(returns[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(returns[0, 1].item(), 0.0, places=6)

    def test_gae_discounted_two_step(self):
        # γ=0.5, λ=0.5，无终止：
        # δ_1 = r1 - V1 = 1；A_1 = 1
        # δ_0 = r0 + γ V1 - V0 = 1；A_0 = 1 + γλ A_1 = 1 + 0.25 = 1.25
        rewards = torch.tensor([[1.0, 1.0]])
        values = torch.tensor([[0.0, 0.0]])
        dones = torch.tensor([[0.0, 0.0]])
        adv, returns = compute_gae(rewards, values, dones, gamma=0.5, lam=0.5)

        self.assertAlmostEqual(adv[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(adv[0, 0].item(), 1.25, places=6)
        self.assertAlmostEqual(returns[0, 0].item(), 1.25, places=6)
        self.assertAlmostEqual(returns[0, 1].item(), 1.0, places=6)

    def test_gae_nonterminal_mask_blocks_backprop(self):
        # d_0=1：t=0 已终止，A_0 不应吸收 t=1 的优势
        rewards = torch.tensor([[0.0, 1.0]])
        values = torch.tensor([[0.0, 0.0]])
        dones = torch.tensor([[1.0, 0.0]])
        adv, _ = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        self.assertAlmostEqual(adv[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[0, 1].item(), 1.0, places=6)


class TestClippedPolicyLoss(unittest.TestCase):
    def test_positive_advantage_clips_upper_ratio(self):
        # A=1>0，ρ=1.5 > 1+ε=1.2；min(1.5, 1.2)=1.2，损失 -1.2
        new_logp = torch.tensor([[torch.log(torch.tensor(1.5)).item()]])
        old_logp = torch.zeros_like(new_logp)
        adv = torch.tensor([[1.0]])
        mask = torch.ones_like(adv, dtype=torch.bool)
        loss, ratio = clipped_policy_loss(
            new_logp, old_logp, adv, mask, clip_ratio=0.2
        )
        self.assertAlmostEqual(ratio.item(), 1.5, places=6)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)

    def test_negative_advantage_clips_lower_ratio(self):
        # A=-1<0，ρ=0.5 < 1-ε=0.8；surr1=-0.5，surr2=-0.8，min=-0.8，损失 0.8
        new_logp = torch.tensor([[torch.log(torch.tensor(0.5)).item()]])
        old_logp = torch.zeros_like(new_logp)
        adv = torch.tensor([[-1.0]])
        mask = torch.ones_like(adv, dtype=torch.bool)
        loss, ratio = clipped_policy_loss(
            new_logp, old_logp, adv, mask, clip_ratio=0.2
        )
        self.assertAlmostEqual(ratio.item(), 0.5, places=6)
        self.assertAlmostEqual(loss.item(), 0.8, places=6)

    def test_ratio_in_range_no_clip(self):
        # ρ=1.0，在裁剪区间内，min 退化为 ρ·A
        new_logp = torch.zeros(1, 2)
        old_logp = torch.zeros(1, 2)
        adv = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(adv, dtype=torch.bool)
        loss, _ = clipped_policy_loss(new_logp, old_logp, adv, mask, 0.2)
        # 两条 surrogate：-min(1,1)*1 = -1；-min(1,1)*(-1) = 1；平均 0
        self.assertAlmostEqual(loss.item(), 0.0, places=6)


class TestClippedValueLoss(unittest.TestCase):
    def test_no_drift_matches_mse(self):
        values = torch.tensor([[0.0]])
        old_values = torch.tensor([[0.0]])
        returns = torch.tensor([[1.0]])
        mask = torch.ones_like(returns, dtype=torch.bool)
        vloss = clipped_value_loss(values, old_values, returns, mask, 0.2)
        self.assertAlmostEqual(vloss.item(), 1.0, places=6)

    def test_clipped_takes_max(self):
        # 新价值偏差超出裁剪区间，裁剪侧平方误差更大，取 max
        values = torch.tensor([[2.0]])
        old_values = torch.tensor([[0.0]])
        returns = torch.tensor([[0.0]])
        mask = torch.ones_like(returns, dtype=torch.bool)
        vloss = clipped_value_loss(values, old_values, returns, mask, 0.2)
        # 未裁剪：(2-0)^2=4；裁剪后 V^c=0.2，(0.2-0)^2=0.04；max=4
        self.assertAlmostEqual(vloss.item(), 4.0, places=6)


class TestPPOCompositeLoss(unittest.TestCase):
    def test_composite_includes_kl(self):
        torch.manual_seed(0)
        B, T = 2, 3
        new_logp = torch.randn(B, T)
        old_logp = torch.randn(B, T)
        ref_logp = torch.randn(B, T)
        adv = torch.randn(B, T)
        values = torch.randn(B, T)
        old_values = torch.randn(B, T)
        returns = adv + values
        mask = torch.ones(B, T, dtype=torch.bool)

        out = ppo_loss(
            new_logp, old_logp, adv, values, old_values, returns, mask,
            ref_logps=ref_logp, kl_coef=0.05, kl_estimator="k3",
        )
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["kl"]))
        # k3 估计量恒非负
        self.assertGreaterEqual(out["kl"].item(), -1e-6)

    def test_no_ref_no_kl(self):
        new_logp = torch.randn(1, 2)
        old_logp = torch.randn(1, 2)
        adv = torch.randn(1, 2)
        values = torch.randn(1, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        out = ppo_loss(
            new_logp, old_logp, adv, values, values, adv + values, mask,
            ref_logps=None, kl_coef=0.05,
        )
        self.assertAlmostEqual(out["kl"].item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

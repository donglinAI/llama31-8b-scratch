"""DPO 损失数值与对称性测试（Rafailov et al., 2023）。"""

import math
import unittest

import torch
import torch.nn.functional as F

from llama31_8b_scratch.align.dpo import dpo_loss, implicit_reward


class TestDPOLoss(unittest.TestCase):
    def test_zero_delta_loss_is_log2(self):
        # 策略与参考模型完全一致 -> 隐式奖励差为 0 -> L = -log σ(0) = log 2
        same = torch.tensor([-1.0, -2.0])
        out = dpo_loss(
            policy_chosen_logps=same,
            policy_rejected_logps=torch.tensor([-3.0, -4.0]),
            ref_chosen_logps=same,
            ref_rejected_logps=torch.tensor([-3.0, -4.0]),
            beta=1.0,
        )
        self.assertAlmostEqual(out["loss"].item(), math.log(2.0), places=6)

    def test_closed_form_logits(self):
        # β=1：chosen Δ=(-1)-(-2)=1，rejected Δ=(-3)-(-2)=-1，logit=2
        out = dpo_loss(
            policy_chosen_logps=torch.tensor([-1.0]),
            policy_rejected_logps=torch.tensor([-3.0]),
            ref_chosen_logps=torch.tensor([-2.0]),
            ref_rejected_logps=torch.tensor([-2.0]),
            beta=1.0,
        )
        expected = -F.logsigmoid(torch.tensor(2.0)).item()
        self.assertAlmostEqual(out["loss"].item(), expected, places=6)
        self.assertAlmostEqual(out["logits"].item(), 2.0, places=6)
        self.assertAlmostEqual(out["rewards_chosen"].item(), 1.0, places=6)
        self.assertAlmostEqual(out["rewards_rejected"].item(), -1.0, places=6)

    def test_beta_scales_rewards(self):
        out = dpo_loss(
            policy_chosen_logps=torch.tensor([-1.0]),
            policy_rejected_logps=torch.tensor([-3.0]),
            ref_chosen_logps=torch.tensor([-2.0]),
            ref_rejected_logps=torch.tensor([-2.0]),
            beta=0.1,
        )
        # 隐式奖励 = β · Δ
        self.assertAlmostEqual(out["rewards_chosen"].item(), 0.1, places=6)
        self.assertAlmostEqual(out["rewards_rejected"].item(), -0.1, places=6)

    def test_swap_symmetry(self):
        # 交换 chosen/rejected 后 DPO logit 取反，损失变为 -log σ(-z)
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.0])
        out_a = dpo_loss(pc, pr, rc, rr, beta=1.0)
        out_b = dpo_loss(pr, pc, rr, rc, beta=1.0)
        self.assertAlmostEqual(out_a["logits"].item(), 2.0, places=6)
        self.assertAlmostEqual(out_b["logits"].item(), -2.0, places=6)
        self.assertAlmostEqual(
            out_b["loss"].item(),
            -F.logsigmoid(torch.tensor(-2.0)).item(),
            places=6,
        )
        # 排序准确率方向相反
        self.assertEqual(out_a["accuracy"].item(), 1.0)
        self.assertEqual(out_b["accuracy"].item(), 0.0)

    def test_label_smoothing(self):
        out = dpo_loss(
            policy_chosen_logps=torch.tensor([-1.0]),
            policy_rejected_logps=torch.tensor([-3.0]),
            ref_chosen_logps=torch.tensor([-2.0]),
            ref_rejected_logps=torch.tensor([-2.0]),
            beta=1.0,
            label_smoothing=0.1,
        )
        z = 2.0
        expected = -(0.9 * F.logsigmoid(torch.tensor(z))
                     + 0.1 * F.logsigmoid(torch.tensor(-z))).item()
        self.assertAlmostEqual(out["loss"].item(), expected, places=6)

    def test_implicit_reward(self):
        r = implicit_reward(torch.tensor([-1.0]), torch.tensor([-2.0]), beta=0.1)
        self.assertAlmostEqual(r.item(), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()

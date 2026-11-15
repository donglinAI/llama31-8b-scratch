"""ORPO 损失组成测试（Hong & Lee, 2024）。"""

import math
import unittest

import torch
import torch.nn.functional as F

from llama31_8b_scratch.align.orpo import log_odds, orpo_loss, sequence_mean_logprobs


def _ref_log_odds(lp: float) -> float:
    p = math.exp(lp)
    return lp - math.log1p(-p)


class TestLogOdds(unittest.TestCase):
    def test_log_odds_matches_closed_form(self):
        lp = torch.tensor([-0.5, -1.0])
        odds = log_odds(lp)
        self.assertAlmostEqual(odds[0].item(), _ref_log_odds(-0.5), places=6)
        self.assertAlmostEqual(odds[1].item(), _ref_log_odds(-1.0), places=6)

    def test_sequence_mean_logprobs(self):
        token_logps = torch.tensor([[-0.2, -0.4], [-0.1, -0.3]])
        mask = torch.tensor([[True, True], [True, False]])
        mean_lp = sequence_mean_logprobs(token_logps, mask)
        self.assertAlmostEqual(mean_lp[0].item(), -0.3, places=6)
        self.assertAlmostEqual(mean_lp[1].item(), -0.1, places=6)


class TestORPOLoss(unittest.TestCase):
    def test_loss_composition(self):
        chosen_lp = torch.tensor([-0.5])
        rejected_lp = torch.tensor([-1.0])
        chosen_nll = torch.tensor(0.5)
        beta = 1.0
        lambda_or = 1.0

        out = orpo_loss(chosen_lp, rejected_lp, chosen_nll,
                        beta=beta, lambda_or=lambda_or)

        odds_w = _ref_log_odds(-0.5)
        odds_l = _ref_log_odds(-1.0)
        or_logit = beta * (odds_w - odds_l)
        expected_or = -F.logsigmoid(torch.tensor(or_logit)).item()

        self.assertAlmostEqual(out["sft_loss"].item(), 0.5, places=6)
        self.assertAlmostEqual(out["or_loss"].item(), expected_or, places=6)
        # total = SFT + λ_OR · OR
        expected_total = 0.5 + lambda_or * expected_or
        self.assertAlmostEqual(out["loss"].item(), expected_total, places=6)
        self.assertEqual(out["accuracy"].item(), 1.0)

    def test_lambda_or_scales_or_term(self):
        chosen_lp = torch.tensor([-0.5])
        rejected_lp = torch.tensor([-1.0])
        nll = torch.tensor(0.5)
        out = orpo_loss(chosen_lp, rejected_lp, nll, beta=1.0, lambda_or=0.1)
        expected_total = 0.5 + 0.1 * out["or_loss"].item()
        self.assertAlmostEqual(out["loss"].item(), expected_total, places=6)

    def test_worse_ordering_accuracy_zero(self):
        # chosen 平均对数概率低于 rejected -> 赔率比为负
        out = orpo_loss(
            torch.tensor([-1.0]), torch.tensor([-0.5]),
            torch.tensor(0.5), beta=1.0, lambda_or=0.1,
        )
        self.assertEqual(out["accuracy"].item(), 0.0)
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()

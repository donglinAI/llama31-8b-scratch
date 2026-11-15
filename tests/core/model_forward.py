"""整模型前向 / 反向形状、梯度与奖励头测试（CPU）。"""

import unittest

import torch

from llama31_8b_scratch.model.causal_lm import ScratchForCausalLM
from llama31_8b_scratch.model.sequence import ScratchForSequenceClassification
from tests.core._scratch_test_utils import build_causal_lm, smoke_config


class TestModelForwardBackward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_causal_lm()
        self.ids = torch.randint(0, 64, (2, 5))
        self.labels = self.ids.clone()

    def test_forward_logits_shape(self):
        out = self.model(self.ids)
        self.assertEqual(out.logits.shape, (2, 5, 64))
        self.assertIsNone(out.loss)

    def test_loss_is_finite_scalar(self):
        out = self.model(self.ids, labels=self.labels)
        self.assertEqual(out.loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(out.loss))

    def test_backward_populates_gradients(self):
        out = self.model(self.ids, labels=self.labels)
        out.loss.backward()

        # 嵌入层、某层注意力投影、输出头都应有梯度
        embed_grad = self.model.model.embed_tokens.weight.grad
        self.assertIsNotNone(embed_grad)
        self.assertEqual(embed_grad.shape, self.model.model.embed_tokens.weight.shape)

        q_grad = self.model.model.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(q_grad)
        self.assertEqual(
            q_grad.shape, self.model.model.layers[0].self_attn.q_proj.weight.shape
        )

        head_grad = self.model.lm_head.weight.grad
        self.assertIsNotNone(head_grad)
        self.assertEqual(head_grad.shape, self.model.lm_head.weight.shape)

    def test_ignored_label_does_not_crash(self):
        labels = torch.full_like(self.labels, -100)
        out = self.model(self.ids, labels=labels)
        self.assertEqual(out.logits.shape, (2, 5, 64))

    def test_tied_word_embeddings(self):
        model = ScratchForCausalLM(smoke_config(tie_word_embeddings=True))
        out = model(self.ids)
        self.assertEqual(out.logits.shape, (2, 5, 64))
        self.assertIsNone(model.lm_head)
        state = model.state_dict()
        self.assertNotIn("lm_head.weight", state)
        self.assertIn("model.embed_tokens.weight", state)


class TestSequenceClassificationHead(unittest.TestCase):
    def test_pooled_score_shape(self):
        model = ScratchForSequenceClassification(smoke_config(), num_labels=1)
        ids = torch.randint(0, 64, (3, 5))
        out = model(ids)
        self.assertEqual(out.logits.shape, (3, 1))
        self.assertEqual(model.score.weight.shape, (1, 32))

    def test_score_pair_returns_two_rewards(self):
        model = ScratchForSequenceClassification(smoke_config(), num_labels=1)
        chosen = torch.randint(0, 64, (2, 5))
        rejected = torch.randint(0, 64, (2, 5))
        r_chosen, r_rejected = model.score_pair(chosen, rejected)
        self.assertEqual(r_chosen.shape, (2, 1))
        self.assertEqual(r_rejected.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()

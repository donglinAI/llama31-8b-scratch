"""RoPE 形状、旋转性质与 scaling 测试（CPU）。"""

import unittest

import torch

from llama31_8b_scratch.model.rope import (
    ScratchRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from tests.core._scratch_test_utils import smoke_config


class TestRotateHalf(unittest.TestCase):
    def test_known_mapping(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
        self.assertTrue(torch.allclose(rotate_half(x), expected))

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 5, 8)
        self.assertEqual(rotate_half(x).shape, x.shape)


class TestScratchRotaryEmbedding(unittest.TestCase):
    def test_cos_sin_shape(self):
        emb = ScratchRotaryEmbedding(smoke_config())
        q = torch.randn(2, 4, 5, 8)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos, sin = emb(q, position_ids)
        self.assertEqual(cos.shape, (2, 1, 5, 8))
        self.assertEqual(sin.shape, (2, 1, 5, 8))

    def test_rotation_preserves_vector_norm(self):
        emb = ScratchRotaryEmbedding(smoke_config())
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 2, 5, 8)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos, sin = emb(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        # 旋转是正交变换，不改变向量模长
        self.assertTrue(torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5))

    def test_linear_scaling_divides_inverse_freq(self):
        cfg = smoke_config(rope_scaling={"type": "linear", "factor": 4.0})
        emb = ScratchRotaryEmbedding(cfg)
        idx = torch.arange(0, 8, 2, dtype=torch.float32)
        base = 1.0 / (10000.0 ** (idx / 8))
        expected = base / 4.0
        self.assertTrue(torch.allclose(emb.inv_freq, expected, atol=1e-8))

    def test_dynamic_ntk_scaling(self):
        cfg = smoke_config(rope_scaling={"type": "dynamic", "factor": 4.0})
        emb = ScratchRotaryEmbedding(cfg)
        idx = torch.arange(0, 8, 2, dtype=torch.float32)
        base = 1.0 / (10000.0 ** (idx / 8))
        expected = base * (4.0 ** (idx / 8))
        self.assertTrue(torch.allclose(emb.inv_freq, expected, atol=1e-8))

    def test_apply_rotary_output_shapes(self):
        emb = ScratchRotaryEmbedding(smoke_config())
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 2, 5, 8)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos, sin = emb(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)


if __name__ == "__main__":
    unittest.main()

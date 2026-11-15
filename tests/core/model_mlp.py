"""ScratchMLP（SwiGLU）形状与数值测试（CPU）。"""

import unittest

import torch
from torch.nn import functional as F

from llama31_8b_scratch.model.mlp import ScratchMLP
from tests.core._scratch_test_utils import smoke_config


class TestScratchMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = ScratchMLP(smoke_config()).eval()
        self.x = torch.randn(2, 5, 32)

    def test_output_shape(self):
        out = self.mlp(self.x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_swiglu_manual_forward(self):
        # 按公式手算：down( silu(gate x) ⊙ up x )，与模块前向对齐
        gate = F.silu(self.mlp.gate_proj(self.x))
        up = self.mlp.up_proj(self.x)
        expected = self.mlp.down_proj(gate * up)
        actual = self.mlp(self.x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_gate_up_down_projection_shapes(self):
        self.assertEqual(self.mlp.gate_proj.weight.shape, (64, 32))
        self.assertEqual(self.mlp.up_proj.weight.shape, (64, 32))
        self.assertEqual(self.mlp.down_proj.weight.shape, (32, 64))

    def test_no_bias_in_projections(self):
        self.assertIsNone(self.mlp.gate_proj.bias)
        self.assertIsNone(self.mlp.up_proj.bias)
        self.assertIsNone(self.mlp.down_proj.bias)


if __name__ == "__main__":
    unittest.main()

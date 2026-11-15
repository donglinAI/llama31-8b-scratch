"""ScratchRMSNorm 数值与形状测试（CPU）。"""

import unittest

import torch

from llama31_8b_scratch.model.norm import ScratchRMSNorm


class TestScratchRMSNorm(unittest.TestCase):
    def test_output_shape_matches_input(self):
        norm = ScratchRMSNorm(hidden_size=8, eps=1e-5)
        x = torch.randn(3, 5, 8)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_constant_vector_normalized_to_unit_rms(self):
        # 全 1 向量：mean(x^2)=1，归一化后近似全 1（weight 初始为 1）
        norm = ScratchRMSNorm(hidden_size=4, eps=1e-5)
        x = torch.ones(2, 4)
        out = norm(x)
        self.assertTrue(torch.allclose(out, torch.ones_like(out), atol=1e-4))

    def test_known_value_two_dimensions(self):
        # x=[3,4]：mean(x^2)=(9+16)/2=12.5，RMS=sqrt(12.5+eps)
        eps = 1e-5
        norm = ScratchRMSNorm(hidden_size=2, eps=eps)
        x = torch.tensor([[3.0, 4.0]])
        out = norm(x)
        rms = (12.5 + eps) ** 0.5
        expected = torch.tensor([[3.0 / rms, 4.0 / rms]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_weight_scales_output(self):
        norm = ScratchRMSNorm(hidden_size=2, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, 3.0]))
        x = torch.tensor([[3.0, 4.0]])
        out = norm(x)
        rms = (12.5 + 1e-5) ** 0.5
        expected = torch.tensor([[2.0 * 3.0 / rms, 3.0 * 4.0 / rms]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_dtype_preserved(self):
        norm = ScratchRMSNorm(hidden_size=4, eps=1e-5)
        x = torch.randn(2, 4, dtype=torch.float32)
        self.assertEqual(norm(x).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

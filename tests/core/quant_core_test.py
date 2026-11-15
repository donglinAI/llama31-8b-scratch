"""手写量化数学单测：对称/仿射、roundtrip、误差边界、zero-point。"""

from __future__ import annotations

import unittest

import numpy as np

from llama31_8b_scratch.quantization.core import (
    affine_dequantize,
    affine_quantize,
    int_range,
    quantization_error,
    symmetric_dequantize,
    symmetric_quantize,
)


class TestIntRange(unittest.TestCase):
    def test_int8(self):
        self.assertEqual(int_range(8), (-128, 127))

    def test_int4(self):
        self.assertEqual(int_range(4), (-8, 7))

    def test_unsigned(self):
        self.assertEqual(int_range(4, signed=False), (0, 15))


class TestSymmetricQuant(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.standard_normal(512) * 2.0

    def test_roundtrip_int8(self):
        r = symmetric_quantize(self.x, num_bits=8)
        xh = symmetric_dequantize(r)
        # 误差不超过 scale/2
        err = quantization_error(self.x, xh)
        self.assertLessEqual(err["max_abs"], r.scale / 2 + 1e-9)
        # zero_point 必须为 0
        self.assertEqual(r.zero_point, 0)
        # 整数落在 [-128,127]
        self.assertGreaterEqual(int(r.q.min()), -128)
        self.assertLessEqual(int(r.q.max()), 127)

    def test_roundtrip_int4(self):
        r = symmetric_quantize(self.x, num_bits=4)
        xh = r.dequantize()
        err = quantization_error(self.x, xh)
        self.assertLessEqual(err["max_abs"], r.scale / 2 + 1e-9)
        self.assertEqual(r.zero_point, 0)
        # int4 范围 [-8, 7]
        self.assertGreaterEqual(int(r.q.min()), -8)
        self.assertLessEqual(int(r.q.max()), 7)

    def test_zero_tensor(self):
        x = np.zeros(16)
        r = symmetric_quantize(x, num_bits=8)
        xh = r.dequantize()
        np.testing.assert_allclose(xh, 0.0, atol=1e-9)

    def test_scale_proportional_to_max_abs(self):
        x = np.array([-3.0, 1.0, 2.0])
        r = symmetric_quantize(x, num_bits=8)
        # scale = max_abs / 127 = 3/127
        self.assertAlmostEqual(r.scale, 3.0 / 127.0, places=6)


class TestAffineQuant(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.uniform(-1.0, 1.0, size=1024)

    def test_roundtrip_int8(self):
        r = affine_quantize(self.x, num_bits=8)
        xh = affine_dequantize(r)
        err = quantization_error(self.x, xh)
        self.assertLessEqual(err["max_abs"], r.scale / 2 + 1e-9)

    def test_roundtrip_int4(self):
        r = affine_quantize(self.x, num_bits=4)
        xh = r.dequantize()
        err = quantization_error(self.x, xh)
        self.assertLessEqual(err["max_abs"], r.scale / 2 + 1e-9)
        self.assertGreaterEqual(int(r.q.min()), -8)
        self.assertLessEqual(int(r.q.max()), 7)

    def test_zero_point_maps_xmin_to_qmin(self):
        """zero_point 的设计目标：反量化后 q_min 对应的值接近 x_min。"""
        x = np.array([-5.0, -2.0, 0.0, 3.0])
        r = affine_quantize(x, num_bits=8)
        xh = r.dequantize()
        # q_min 对应的反量化值应接近 x_min
        reconstructed_at_qmin = r.scale * (r.q.min() - r.zero_point)
        self.assertAlmostEqual(reconstructed_at_qmin, x.min(), delta=r.scale)

    def test_constant_tensor(self):
        x = np.full(64, 3.5)
        r = affine_quantize(x, num_bits=8)
        xh = r.dequantize()
        # 常数张量可以被精确表示（量化到 0 误差以内）
        self.assertLessEqual(quantization_error(x, xh)["max_abs"], 1e-6)


class TestErrorBounds(unittest.TestCase):
    def test_int4_error_larger_than_int8(self):
        """同一张量：int4 相对误差应显著大于 int8（这是为什么需要 GPTQ/AWQ）。"""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(2048)
        r8 = symmetric_quantize(x, num_bits=8)
        r4 = symmetric_quantize(x, num_bits=4)
        e8 = quantization_error(x, r8.dequantize())
        e4 = quantization_error(x, r4.dequantize())
        self.assertGreater(e4["relative_l2"], e8["relative_l2"])


if __name__ == "__main__":
    unittest.main()

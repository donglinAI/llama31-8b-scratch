"""评测指标聚合单测：均值、准确率、置信区间、正态分位数。"""

from __future__ import annotations

import math
import unittest

from llama31_8b_scratch.evaluation.metrics_core import (
    accuracy,
    confidence_interval,
    mean,
    normal_ppf,
    std,
)


class TestBasicStats(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(mean([-1.0, 1.0]), 0.0)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_sample_std(self):
        # 样本 [2,4,4,4,5,5,7,9]：mean=5，平方偏差和=32，样本方差=32/7
        self.assertAlmostEqual(
            std([2, 4, 4, 4, 5, 5, 7, 9]),
            (32 / 7) ** 0.5,
            places=6,
        )

    def test_population_std(self):
        # 总体标准差 ddof=0：方差=32/8=4，std=2.0
        self.assertAlmostEqual(
            std([2, 4, 4, 4, 5, 5, 7, 9], ddof=0),
            2.0,
            places=6,
        )


class TestAccuracy(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(accuracy([0, 1, 2], [0, 1, 2]), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(accuracy([0, 1, 2, 0], [0, 1, 1, 0]), 0.75)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            accuracy([0, 1], [0])


class TestNormalPpf(unittest.TestCase):
    def test_known_values(self):
        # 标准正态分位数
        self.assertAlmostEqual(normal_ppf(0.5), 0.0, places=9)
        self.assertAlmostEqual(normal_ppf(0.975), 1.959963985, places=6)
        self.assertAlmostEqual(normal_ppf(0.025), -1.959963985, places=6)
        self.assertAlmostEqual(normal_ppf(0.995), 2.5758293, places=5)

    def test_domain(self):
        with self.assertRaises(ValueError):
            normal_ppf(0.0)
        with self.assertRaises(ValueError):
            normal_ppf(1.0)


class TestConfidenceInterval(unittest.TestCase):
    def test_ci_contains_mean(self):
        samples = [0.5, 0.55, 0.48, 0.52, 0.5, 0.49, 0.51]
        mu, lo, hi = confidence_interval(samples, 0.95)
        self.assertLess(lo, mu)
        self.assertLess(mu, hi)
        # 均值就在区间内
        self.assertAlmostEqual(mu, mean(samples), places=9)

    def test_small_n_raises(self):
        with self.assertRaises(ValueError):
            confidence_interval([1.0])


if __name__ == "__main__":
    unittest.main()

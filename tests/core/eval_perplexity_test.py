"""困惑度单测：用闭式分布验证 CE→NLL→PPL 的数值正确性。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from llama31_8b_scratch.evaluation.perplexity import (
    corpus_ppl,
    cross_entropy,
    log_softmax,
    nll,
    perplexity,
)


class TestLogSoftmax(unittest.TestCase):
    def test_uniform_logits_give_uniform_prob(self):
        # 全 0 logits → softmax 是均匀分布
        logits = np.zeros((1, 5))
        logp = log_softmax(logits, axis=-1)
        expected = math.log(1 / 5)
        for v in logp[0]:
            self.assertAlmostEqual(v, expected, places=9)

    def test_sharp_distribution(self):
        # 第 0 个 logit 远大于其它，softmax 后接近 one-hot
        logits = np.array([[100.0, 0.0, 0.0]])
        logp = log_softmax(logits, axis=-1)
        self.assertAlmostEqual(logp[0, 0], 0.0, places=6)
        self.assertLess(logp[0, 1], -90.0)


class TestPerplexityClosedForm(unittest.TestCase):
    def test_uniform_over_V_gives_ppl_equal_V(self):
        """在大小为 V 的词表上均匀预测，PPL 应严格等于 V。"""
        V = 4
        T = 10
        logits = np.zeros((T, V))  # 均匀
        targets = np.arange(T) % V
        ce = cross_entropy(logits, targets)
        # CE = -log(1/V) = log V
        self.assertAlmostEqual(ce, math.log(V), places=9)
        ppl = perplexity(logits, targets)
        self.assertAlmostEqual(ppl, float(V), places=6)

    def test_perfect_prediction_gives_ppl_one(self):
        """one-hot 预测到正确 token，CE=0，PPL=1。"""
        V = 3
        T = 5
        logits = np.full((T, V), -100.0)
        targets = np.array([0, 1, 2, 0, 1])
        for t, v in enumerate(targets):
            logits[t, v] = 100.0  # 正确 token 概率≈1
        ce = cross_entropy(logits, targets)
        self.assertAlmostEqual(ce, 0.0, places=5)
        self.assertAlmostEqual(perplexity(logits, targets), 1.0, places=4)

    def test_nll_alias_equals_cross_entropy(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((8, 6))
        targets = rng.integers(0, 6, size=8)
        self.assertAlmostEqual(
            nll(logits, targets), cross_entropy(logits, targets), places=12
        )

    def test_mask_excludes_padding(self):
        logits = np.zeros((3, 2))  # uniform over 2
        targets = np.array([0, 1, 0])
        mask = np.array([1.0, 1.0, 0.0])  # 第 3 个不计
        # 只在前 2 个位置均匀预测 → CE = log 2
        ce = cross_entropy(logits, targets, mask=mask)
        self.assertAlmostEqual(ce, math.log(2), places=9)


class TestCorpusAggregation(unittest.TestCase):
    def test_token_weighted_not_document_average(self):
        """两篇文档：一篇短但 PPL 高，一篇长且 PPL 低。
        corpus PPL 应按 token 数加权，而不是两篇 PPL 的算术平均。"""
        # 文档 A：T=2，uniform over V=2 → PPL=2
        la = np.zeros((2, 2))
        ta = np.array([0, 1])
        # 文档 B：T=8，uniform over V=4 → PPL=4
        lb = np.zeros((8, 4))
        tb = np.arange(8) % 4
        ppl, total_tokens, mean_nll = corpus_ppl([(la, ta), (lb, tb)])
        self.assertEqual(total_tokens, 10)
        # mean_nll = (2*log2 + 8*log4)/10 = (2*0.693 + 8*1.386)/10
        expected_nll = (2 * math.log(2) + 8 * math.log(4)) / 10
        self.assertAlmostEqual(mean_nll, expected_nll, places=9)
        self.assertAlmostEqual(ppl, math.exp(expected_nll), places=6)


if __name__ == "__main__":
    unittest.main()

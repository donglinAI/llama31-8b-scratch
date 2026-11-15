"""多选 loglikelihood 评测单测。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from llama31_8b_scratch.evaluation.multiple_choice import (
    mc_accuracy,
    mc_choice_logprobs,
    rank_choice,
    sequence_logprob,
)


class TestSequenceLogprob(unittest.TestCase):
    def test_basic(self):
        # 模型对第一个 token 给 0.8/0.1/0.1，第二个 token 给 0.1/0.8/0.1
        logits = np.log(np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ]))
        ids = np.array([0, 1])  # 第一个选 0，第二个选 1
        # log p = log 0.8 + log 0.8
        expected = 2 * math.log(0.8)
        self.assertAlmostEqual(sequence_logprob(logits, ids), expected, places=6)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            sequence_logprob(np.zeros((3, 4)), np.array([0, 1]))


class TestMcChoice(unittest.TestCase):
    def test_picks_highest_score(self):
        # 三个选项：模型明显偏向选项 2
        c1 = np.log(np.array([[0.7, 0.2, 0.1]]))
        c2 = np.log(np.array([[0.1, 0.8, 0.1]]))
        c3 = np.log(np.array([[0.2, 0.3, 0.5]]))
        scores = mc_choice_logprobs(
            [c1, c2, c3],
            [np.array([0]), np.array([1]), np.array([2])],
        )
        self.assertEqual(rank_choice(scores), 1)

    def test_mc_accuracy(self):
        records = [
            {"scores": [-1.0, -2.0, -3.0, -4.0], "gold": 0},  # 对
            {"scores": [-2.0, -1.0, -3.0, -4.0], "gold": 0},  # 错（应选 1）
            {"scores": [-4.0, -3.0, -2.0, -1.0], "gold": 3},  # 对
        ]
        self.assertAlmostEqual(mc_accuracy(records), 2 / 3, places=9)


if __name__ == "__main__":
    unittest.main()

"""生成式评测框架单测。"""

from __future__ import annotations

import unittest

from llama31_8b_scratch.evaluation.generation_eval import (
    GenSample,
    GenerationEval,
    aggregate_scores,
    contains_match,
    exact_match,
    regex_match,
)


class TestScorers(unittest.TestCase):
    def test_exact_match(self):
        s = GenSample(prompt="", generated=" 42\n", reference="42")
        self.assertEqual(exact_match(s), 1.0)
        s2 = GenSample(prompt="", generated="43", reference="42")
        self.assertEqual(exact_match(s2), 0.0)

    def test_regex_match(self):
        s = GenSample(
            prompt="",
            generated="The answer is 42.",
            reference="42",
        )
        self.assertEqual(regex_match(s, r"(-?\d+)"), 1.0)
        s2 = GenSample(prompt="", generated="no number here", reference="42")
        self.assertEqual(regex_match(s2, r"(-?\d+)"), 0.0)

    def test_contains(self):
        s = GenSample(prompt="", generated="we should use RMSNorm here", reference="RMSNorm")
        self.assertEqual(contains_match(s), 1.0)


class TestAggregate(unittest.TestCase):
    def test_empty(self):
        out = aggregate_scores([])
        self.assertEqual(out["n"], 0)

    def test_mean_and_ci(self):
        out = aggregate_scores([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(out["mean"], 1.0)
        self.assertEqual(out["ci_low"], out["ci_high"])  # 零方差


class TestFramework(unittest.TestCase):
    def test_run_multiple_scorers(self):
        ev = GenerationEval()
        ev.add_scorer("em", exact_match)
        ev.add_scorer("math", lambda s: regex_match(s, r"(-?\d+)"))
        samples = [
            GenSample(prompt="", generated="42", reference="42"),
            GenSample(prompt="", generated="answer: 7", reference="7"),
            GenSample(prompt="", generated="foo", reference="1"),
        ]
        result = ev.run(samples)
        self.assertEqual(set(result.keys()), {"em", "math"})
        # em: 只有第一条 "42"=="42"，第二条 strip 后是 "answer: 7" != "7"
        self.assertAlmostEqual(result["em"]["mean"], 1 / 3, places=6)
        # math: 第一条抽 42==42，第二条抽 7==7，第三条无数字
        self.assertAlmostEqual(result["math"]["mean"], 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()

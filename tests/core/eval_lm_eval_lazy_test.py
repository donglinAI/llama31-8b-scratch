"""lm-eval-harness 惰性封装测试（环境未装 lm_eval）。"""

from __future__ import annotations

import unittest

from llama31_8b_scratch.evaluation.lm_eval_harness import (
    COMMON_TASKS,
    list_available_tasks,
    run_lm_eval,
)


class TestLmEvalLazy(unittest.TestCase):
    def test_common_task_map_present(self):
        # 任务名速查表必须包含 SPEC 要求的核心任务
        for name in ("mmlu", "ceval", "cmmlu", "gpqa"):
            self.assertIn(name, COMMON_TASKS)

    def test_missing_lm_eval_raises_clear_error(self):
        with self.assertRaises(ImportError) as ctx:
            list_available_tasks()
        self.assertIn("lm-evaluation-harness", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))

    def test_run_lm_eval_missing_raises(self):
        with self.assertRaises(ImportError):
            run_lm_eval("fake-model", ["mmlu"], device="cpu")


if __name__ == "__main__":
    unittest.main()

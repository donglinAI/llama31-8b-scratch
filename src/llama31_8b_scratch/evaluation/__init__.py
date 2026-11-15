"""评测子包：困惑度、多选 loglikelihood、生成式打分与 lm-eval-harness 封装。

子包设计分两层：

1. **手写教学层**（纯标准库 + numpy，可选 torch）：
   - :mod:`llama31_8b_scratch.evaluation.metrics_core`  聚合指标（均值、准确率、置信区间）
   - :mod:`llama31_8b_scratch.evaluation.perplexity`   CE→NLL→PPL 的无跳步实现
   - :mod:`llama31_8b_scratch.evaluation.multiple_choice`  loglikelihood 多选评测
   - :mod:`llama31_8b_scratch.evaluation.generation_eval`  生成式评测打分框架
2. **工业封装层**（惰性导入 ``lm_eval``，缺失时给出清晰提示）：
   - :mod:`llama31_8b_scratch.evaluation.lm_eval_harness`  CMMLU / C-Eval / MMLU / GPQA 等任务

模块顶层不 import 任何可选重依赖，保证 ``import llama31_8b_scratch.evaluation`` 在
只装了 torch+numpy+yaml 的环境下也能成功。
"""

from __future__ import annotations

# 只把子模块 import 进来（不 re-export 子模块内部的函数），避免函数名与
# 子模块名互相遮蔽。需要具体函数时用完整路径：
#   from llama31_8b_scratch.evaluation.perplexity import perplexity, cross_entropy
from llama31_8b_scratch.evaluation import (  # noqa: F401
    generation_eval,
    lm_eval_harness,
    metrics_core,
    multiple_choice,
    perplexity,
)

__all__ = [
    "metrics_core",
    "perplexity",
    "multiple_choice",
    "generation_eval",
    "lm_eval_harness",
]

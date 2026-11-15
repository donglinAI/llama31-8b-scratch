"""生成式评测（gen 范式）的打分框架。

适用场景
========
当任务不能用 loglikelihood 一次判对错时（数学题、代码题、开放问答、指令遵循），
需要让模型真的生成一段文本，再用某种"打分器"判分。本模块提供一个最小但
可扩展的打分框架：

1. **规则打分器**（零依赖，CPU 可跑）：
   - ``exact_match``：生成文本 strip 后与参考完全相等；
   - ``regex_match``：用正则从生成文本里抽出答案，再与参考比对；
   - ``contains``：参考字符串是否出现在生成文本里（粗粒度，慎用）。
2. **可插拔 judge**：调用方可以传入任意 ``Callable[[sample], float]``，
   例如用另一个 LLM 当裁判（GPT-as-judge），或者调用外部工具（单元测试、
   编译器）。框架不绑死任何 judge。

为什么不直接用 evaluate / lm-eval 的 metrics
--------------------------------------------
- 教学目的：把"生成 → 抽取 → 比对 → 聚合"这条链路讲清楚；
- 零依赖：本模块只用标准库，单测不需要下载任何数据集；
- 工业落地：真要跑 MMLU/GPQA 时还是走 :mod:`lm_eval_harness`，本模块
  主要服务于自定义生成任务与 smoke 评测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from llama31_8b_scratch.evaluation.metrics_core import (
    confidence_interval,
    mean,
)

__all__ = [
    "GenSample",
    "exact_match",
    "regex_match",
    "contains_match",
    "aggregate_scores",
    "GenerationEval",
]


@dataclass
class GenSample:
    """一条生成式评测样本。"""

    prompt: str
    generated: str
    reference: Optional[str] = None
    extra: Dict = field(default_factory=dict)


def exact_match(sample: GenSample, *, normalize: bool = True) -> float:
    """生成文本与参考完全相等（可选择 strip + 大小写不敏感的粗归一化）。"""
    if sample.reference is None:
        return 0.0
    pred = sample.generated
    gold = sample.reference
    if normalize:
        pred = pred.strip().lower()
        gold = gold.strip().lower()
    return 1.0 if pred == gold else 0.0


def regex_match(
    sample: GenSample,
    pattern: str,
    *,
    group: int = 0,
    normalize: bool = True,
) -> float:
    """用正则从生成文本里抽出一段，再与参考比对。

    例如数学题答案是数字，pattern 可以是 ``r"(-?\\d+(?:\\.\\d+)?)"``。
    """
    if sample.reference is None:
        return 0.0
    m = re.search(pattern, sample.generated)
    if m is None:
        return 0.0
    pred = m.group(group)
    gold = sample.reference
    if normalize:
        pred = pred.strip().lower()
        gold = str(gold).strip().lower()
    return 1.0 if pred == gold else 0.0


def contains_match(sample: GenSample) -> float:
    """参考字符串是否出现在生成文本里（粗粒度，仅作 smoke 用）。"""
    if sample.reference is None:
        return 0.0
    return 1.0 if sample.reference in sample.generated else 0.0


def aggregate_scores(scores: Sequence[float]) -> Dict[str, float]:
    """把一批样本分聚合为均值与 95% 置信区间。"""
    if not scores:
        return {"mean": 0.0, "n": 0, "ci_low": 0.0, "ci_high": 0.0}
    mu, lo, hi = confidence_interval(scores, confidence=0.95)
    return {
        "mean": float(mu),
        "n": len(scores),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


class GenerationEval:
    """生成式评测器：对每条样本用注册的打分器打分并聚合。

    Usage::

        ev = GenerationEval()
        ev.add_scorer("em", exact_match)
        ev.add_scorer("math", lambda s: regex_match(s, r"(-?\\d+)"))
        results = ev.run(samples)

    每个 scorer 可以返回 0~1 的浮点（部分得分），框架不做截断。
    """

    def __init__(self) -> None:
        self._scorers: Dict[str, Callable[[GenSample], float]] = {}

    def add_scorer(
        self, name: str, fn: Callable[[GenSample], float]
    ) -> "GenerationEval":
        if name in self._scorers:
            raise ValueError(f"打分器已存在: {name}")
        self._scorers[name] = fn
        return self

    def run(
        self, samples: Sequence[GenSample]
    ) -> Dict[str, Dict[str, float]]:
        """对所有样本跑所有打分器，返回 ``{scorer_name: aggregate}``。"""
        per_scorer: Dict[str, List[float]] = {name: [] for name in self._scorers}
        for sample in samples:
            for name, fn in self._scorers.items():
                per_scorer[name].append(float(fn(sample)))
        return {
            name: aggregate_scores(scores)
            for name, scores in per_scorer.items()
        }

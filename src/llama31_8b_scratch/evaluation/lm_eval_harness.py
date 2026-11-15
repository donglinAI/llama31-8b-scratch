"""lm-evaluation-harness（EleutherAI）工业封装。

为什么不手写全部 benchmark
==========================
MMLU / CMMLU / C-Eval / GPQA / HellaSwag / ARC / GSM8K 这些数据集的 prompt
模板、few-shot 拼接、答案归一化、子集聚合口径，都已经被
`EleutherAI/lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
反复打磨并成为社区事实标准。自己重写一遍既容易踩"prompt 不一致"的坑，也无法
和别人的数字对齐。所以本项目的策略是：

- **教学层**：在 :mod:`perplexity` / :mod:`multiple_choice` / :mod:`generation_eval`
  里把"loglikelihood 怎么算、多选用 argmax、gen 怎么打分"讲透；
- **工业层**：本模块惰性封装 ``lm_eval``，直接跑社区标准答案。

本模块**不在顶层 import lm_eval**——它是个可选重依赖。调用真正的评测函数时
才 import；未安装时给出明确的 ``pip install lm-eval`` 提示。

常用任务名（lm-eval v0.4.x 注册名，不要自己瞎拼）
=================================================
+----------------+--------------------------------+------------------+
| 简称           | lm-eval 任务名                 | 范式             |
+================+================================+==================+
| MMLU           | ``mmlu``（57 个子集自动聚合）  | loglikelihood    |
+----------------+--------------------------------+------------------+
| C-Eval         | ``ceval``（中文，52 子集）     | loglikelihood    |
+----------------+--------------------------------+------------------+
| CMMLU          | ``cmmlu``（中文，67 子集）     | loglikelihood    |
+----------------+--------------------------------+------------------+
| GPQA           | ``gpqa_diamond``               | loglikelihood +  |
|                |                                | 可选 gen         |
+----------------+--------------------------------+------------------+
| HellaSwag      | ``hellaswag``                  | loglikelihood    |
+----------------+--------------------------------+------------------+
| ARC-Easy       | ``arc_easy``                   | loglikelihood    |
+----------------+--------------------------------+------------------+
| ARC-Challenge  | ``arc_challenge``              | loglikelihood    |
+----------------+--------------------------------+------------------+
| GSM8K          | ``gsm8k``（主结果用 gen）       | gen              |
+----------------+--------------------------------+------------------+
| Winogrande     | ``winogrande``                 | loglikelihood    |
+----------------+--------------------------------+------------------+

注意：lm-eval 的任务名随版本会漂移；跑之前用 :func:`list_available_tasks`
在本机确认，不要硬编码。
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger(__name__)

# 常见任务的"人类可读名 → lm-eval 注册名"映射。
# 仅作速查；真正跑之前应调用 list_available_tasks 校验。
COMMON_TASKS: Dict[str, str] = {
    "mmlu": "mmlu",
    "ceval": "ceval",
    "cmmlu": "cmmlu",
    "gpqa": "gpqa_diamond",
    "hellaswag": "hellaswag",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
}

__all__ = ["COMMON_TASKS", "list_available_tasks", "run_lm_eval"]


def _require_lm_eval():
    """惰性 import lm_eval；未安装时抛出带安装指令的 ImportError。"""
    try:
        import lm_eval  # noqa: F401
        from lm_eval import evaluator
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "未检测到 lm-evaluation-harness。请先安装：\n"
            "    pip install lm-eval[api]>=0.4.4\n"
            "然后再调用本函数。"
        ) from exc
    return lm_eval, evaluator


def list_available_tasks(patterns: Optional[Sequence[str]] = None) -> List[str]:
    """列出本机 lm_eval 已注册的任务名（可传子串过滤）。

    这是个**只读、零成本**的校验函数，强烈建议在真机跑评测前先调用一次，
    确认 ``cmmlu`` / ``ceval`` / ``gpqa_diamond`` 这些名字在当前版本里存在。
    """
    _, evaluator = _require_lm_eval()
    # lm_eval v0.4.x：通过 TaskManager 拿全部任务
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    all_tasks = tm.all_tasks
    if patterns:
        matched = set()
        for p in patterns:
            matched |= {t for t in all_tasks if p in t}
        return sorted(matched)
    return sorted(all_tasks)


def run_lm_eval(
    model_path: str,
    tasks: Sequence[str],
    *,
    model: str = "hf",
    dtype: str = "auto",
    device: str = "cuda",
    num_fewshot: Optional[int] = None,
    batch_size: str | int = 8,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Dict:
    """对 HF 格式模型跑 lm-eval-harness。

    Args:
        model_path: 本地模型目录或 HF hub 模型名（必须包含 config.json +
            safetensors，本项目手写模型保存的目录直接可用）。
        tasks: 任务名列表，例如 ``["mmlu", "ceval", "cmmlu", "gpqa_diamond"]``。
            建议先 :func:`list_available_tasks` 核对。
        model: lm-eval 的模型后端，默认 ``"hf"``（transformers）；
            要走 vLLM 可传 ``"vllm"``。
        dtype: 传给 model_args 的 ``dtype``，默认 ``"auto"``；CPU smoke 用
            ``"float32"``。
        device: ``cuda`` / ``cpu`` / ``auto``。
        num_fewshot: few-shot 数；不传则用每个任务自带的默认值。
        batch_size: 批大小；lm-eval 支持 ``"auto"``。
        limit: 只取每个任务的前 N 条，**仅用于 smoke**；正式评测必须留 None。
        output_path: lm-eval 结果落盘目录（它会自己写 ``results*.json``）。
        trust_remote_code: 是否允许 HF 加载远程代码（自定义模型时用）。

    Returns:
        lm-eval 的结果 dict（``results`` / ``groups`` / ``versions`` 等键）。
        本项目**不在文档里预填任何分数**——真实数字跑完才填。
    """
    _, evaluator = _require_lm_eval()

    model_args_list = [f"pretrained={model_path}", f"dtype={dtype}", f"device={device}"]
    if trust_remote_code:
        model_args_list.append("trust_remote_code=True")
    model_args = ",".join(model_args_list)

    logger.info("lm-eval 启动: model=%s tasks=%s", model, list(tasks))
    logger.info("lm-eval model_args: %s", model_args)

    kwargs: Dict = {
        "model": model,
        "model_args": model_args,
        "tasks": list(tasks),
        "batch_size": batch_size,
    }
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        kwargs["limit"] = limit
        warnings.warn(
            f"lm-eval 正在以 limit={limit} 跑（smoke 模式），正式评测务必置 None",
            stacklevel=2,
        )
    if output_path is not None:
        kwargs["output_path"] = output_path

    results = evaluator.simple_evaluate(**kwargs)
    logger.info("lm-eval 完成。结果键: %s", list(results.keys()))
    return results

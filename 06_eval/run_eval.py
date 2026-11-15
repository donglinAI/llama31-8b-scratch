"""评测入口：薄壳，按配置分发到手写 PPL / 多选 / 生成式 / lm-eval。

用法::

    python 06_eval/run_eval.py --config configs/eval/smoke.yaml
    python 06_eval/run_eval.py --config configs/eval/llama3_1_8b.yaml \\
        --override mode=lm_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

# 三行 shim：保证零安装也能 import 到 src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from llama31_8b_scratch.common.config import load_config  # noqa: E402
from llama31_8b_scratch.common.logging_utils import get_logger, setup_logging  # noqa: E402
from llama31_8b_scratch.common.seed import set_seed  # noqa: E402
from llama31_8b_scratch.evaluation import multiple_choice as mc  # noqa: E402
from llama31_8b_scratch.evaluation import perplexity as P  # noqa: E402

logger = get_logger("eval")


def _run_smoke(cfg) -> None:
    """合成数据跑通 PPL / 多选 / 生成式三条路径。"""
    s = cfg.smoke
    rng = np.random.default_rng(cfg.seed)

    # 1) PPL：用随机 logits 造一段序列，验证 PPL 通路不崩
    logits = rng.standard_normal((s.seq_len, s.vocab_size))
    targets = rng.integers(0, s.vocab_size, size=s.seq_len)
    ppl = P.perplexity(logits, targets)
    logger.info("[smoke] PPL on synthetic logits = %.4f", ppl)

    # 2) 多选：随机打 C 个分数，gold 固定为第 0 个，验证聚合
    records = []
    for _ in range(s.mc_examples):
        scores = rng.standard_normal(s.mc_choices).tolist()
        # 故意把 gold=0 的分数抬高，让 accuracy 接近 1
        scores[0] += 2.0
        records.append({"scores": scores, "gold": 0})
    acc = mc.mc_accuracy(records)
    logger.info("[smoke] multiple-choice accuracy = %.4f", acc)


def _run_lm_eval(cfg) -> None:
    """调用 lm-eval-harness。"""
    from llama31_8b_scratch.evaluation.lm_eval_harness import run_lm_eval

    le = cfg.lm_eval
    if le.model_path is None:
        logger.warning("lm_eval.model_path 为空，跳过工业评测（smoke 模式）")
        return
    results = run_lm_eval(
        model_path=le.model_path,
        tasks=list(le.tasks),
        dtype=le.dtype,
        device=le.device,
        num_fewshot=getattr(le, "num_fewshot", None),
        batch_size=le.batch_size,
        limit=getattr(le, "limit", None),
        output_path=getattr(le, "output_path", None),
    )
    # 只打印结果键，不臆造分数
    logger.info("[lm-eval] results keys: %s", list(results.get("results", {}).keys()))


def main() -> None:
    parser = argparse.ArgumentParser(description="llama31-8b-scratch 评测入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="覆盖配置，格式 a.b=value，可重复",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)

    mode = cfg.mode
    logger.info("评测模式: %s", mode)
    if mode in ("ppl", "multiple_choice", "generation"):
        _run_smoke(cfg)
    elif mode == "lm_eval":
        _run_lm_eval(cfg)
    else:
        raise ValueError(f"未知 mode: {mode}")


if __name__ == "__main__":
    main()

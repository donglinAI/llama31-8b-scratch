#!/usr/bin/env python3
"""GRPO 训练入口（薄封装）。

手写组内相对优势与裁剪策略损失见 :mod:`llama31_8b_scratch.align.grpo`；工业 TRL
``GRPOTrainer`` 惰性封装见 :mod:`llama31_8b_scratch.align.trl_wrappers`。GRPO 无价值网络。

用法::

    python3 05_align/run_grpo.py --config configs/align/grpo_smoke.yaml
    python3 05_align/run_grpo.py --config configs/align/grpo_8b.yaml \
        --override align.group_size=8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from llama31_8b_scratch.align.trl_wrappers import build_trl_wrapper
from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.common.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="GRPO 训练入口")
    parser.add_argument("--config", required=True, help="对齐配置 YAML 路径")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="点号路径覆盖项，可重复传入",
    )
    return parser.parse_args()


def main() -> None:
    """加载配置、构造 TRL 封装并打印训练计划。"""
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    logger = get_logger("llama31_8b_scratch.align.run_grpo")

    method = cfg.get("align.method", "grpo")
    if method != "grpo":
        raise ValueError(f"run_grpo 期望 method=grpo，实际为 {method!r}")

    wrapper = build_trl_wrapper(method, cfg)
    kwargs = wrapper.build_kwargs()

    logger.info("GRPO 配置已加载：%s", args.config)
    logger.info("group_size=%s，clip=%s，kl_coef=%s",
                cfg.get("align.group_size"), cfg.get("align.clip_ratio"),
                cfg.get("align.kl_coef"))
    logger.info("TRL 构造参数：%s", kwargs)

    # TBD（依赖 model/data 代理交付）：
    #   1. 构建策略模型与（可选）参考模型；
    #   2. 对每个 prompt 采样 G 条回答，规则/奖励模型打分；
    #   3. 组内标准化优势后调用 wrapper.create_trainer(...) 训练。
    # 组内优势零均值验证见 tests/core/align_grpo_test.py。


if __name__ == "__main__":
    main()

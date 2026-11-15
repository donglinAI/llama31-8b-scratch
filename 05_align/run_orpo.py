#!/usr/bin/env python3
"""ORPO 训练入口（薄封装）。

手写 SFT 损失 + 赔率比损失见 :mod:`llama31_8b_scratch.align.orpo`；工业 TRL
``ORPOTrainer`` 惰性封装见 :mod:`llama31_8b_scratch.align.trl_wrappers`。ORPO 单模型、
无需参考模型。

用法::

    python3 05_align/run_orpo.py --config configs/align/orpo_smoke.yaml
    python3 05_align/run_orpo.py --config configs/align/orpo_8b.yaml \
        --override align.lambda_or=0.2
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
    parser = argparse.ArgumentParser(description="ORPO 训练入口")
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
    logger = get_logger("llama31_8b_scratch.align.run_orpo")

    method = cfg.get("align.method", "orpo")
    if method != "orpo":
        raise ValueError(f"run_orpo 期望 method=orpo，实际为 {method!r}")

    wrapper = build_trl_wrapper(method, cfg)
    kwargs = wrapper.build_kwargs()

    logger.info("ORPO 配置已加载：%s", args.config)
    logger.info("β=%s，λ_OR=%s", cfg.get("align.beta"),
                cfg.get("align.lambda_or"))
    logger.info("TRL 构造参数：%s", kwargs)

    # TBD（依赖 model/data 代理交付）：
    #   1. 构建单一策略模型（无 ref_model）；
    #   2. 加载 PreferenceDataset；
    #   3. 调用 wrapper.create_trainer(...) 后 trainer.train()。
    # ORPO 损失组成验证见 tests/core/align_orpo_test.py。


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""奖励模型（RM）训练入口（薄封装）。

职责：解析 YAML 配置、设置随机种子、校验方法字段；真实的 backbone 构建、
偏好数据集加载与训练循环对接 model / data 代理交付后接入。手写成对损失见
:mod:`llama31_8b_scratch.align.reward`，数值闭式验证见 ``tests/core/align_reward_test.py``。

用法::

    python3 05_align/run_rm.py --config configs/align/rm_smoke.yaml
    python3 05_align/run_rm.py --config configs/align/rm_8b.yaml \
        --override trainer.batch_size=16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.common.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数：``--config`` 必填，``--override`` 可重复。"""
    parser = argparse.ArgumentParser(description="奖励模型成对训练入口")
    parser.add_argument("--config", required=True, help="对齐配置 YAML 路径")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="点号路径覆盖项，如 trainer.lr=1e-6，可重复传入",
    )
    return parser.parse_args()


def main() -> None:
    """加载配置并打印训练计划；模型/数据对接点见模块 docstring。"""
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 42))
    logger = get_logger("llama31_8b_scratch.align.run_rm")

    method = cfg.get("align.method", "rm")
    if method != "rm":
        raise ValueError(f"run_rm 期望 method=rm，实际为 {method!r}")

    logger.info("奖励模型训练配置已加载：%s", args.config)
    logger.info("数据路径：train=%s", cfg.get("data.train_path"))
    logger.info("成对间隔 margin=%s", cfg.get("align.margin", 0.0))
    logger.info("训练超参：%s", cfg.get("trainer", {}))

    # TBD（依赖 model/data 代理交付）：
    #   1. 用 config.model 构建 backbone 与标量打分头（ScratchForSequenceClassification）；
    #   2. 用 PreferenceDataset / PreferenceCollator 加载偏好对；
    #   3. 前向得到 chosen/rejected 打分，调用 pairwise_loss 反向更新。
    # CPU 闭式数值验证见 tests/core/align_reward_test.py，不在此编造实测指标。


if __name__ == "__main__":
    main()

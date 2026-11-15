#!/usr/bin/env python3
"""PPO（经典 RLHF）训练入口（薄封装）。

手写 GAE / 裁剪策略目标 / 价值裁剪 / KL 惩罚见 :mod:`llama31_8b_scratch.align.ppo`；
工业 TRL ``PPOTrainer`` 惰性封装见 :mod:`llama31_8b_scratch.align.trl_wrappers`。

用法::

    python3 05_align/run_ppo.py --config configs/align/ppo_smoke.yaml
    python3 05_align/run_ppo.py --config configs/align/ppo_8b.yaml \
        --override align.kl_coef=0.1
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
    parser = argparse.ArgumentParser(description="PPO 经典 RLHF 训练入口")
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
    logger = get_logger("llama31_8b_scratch.align.run_ppo")

    method = cfg.get("align.method", "ppo")
    if method != "ppo":
        raise ValueError(f"run_ppo 期望 method=ppo，实际为 {method!r}")

    wrapper = build_trl_wrapper(method, cfg)
    kwargs = wrapper.build_kwargs()

    logger.info("PPO 配置已加载：%s", args.config)
    logger.info("γ=%s，λ=%s，clip=%s，kl_coef=%s",
                cfg.get("align.gamma"), cfg.get("align.lam"),
                cfg.get("align.clip_ratio"), cfg.get("align.kl_coef"))
    logger.info("TRL 构造参数：%s", kwargs)

    # TBD（依赖 model/data 代理交付）：
    #   1. 构建 policy / ref / reward / value 四模型；
    #   2. 闭环：采样 -> RM 打分 -> GAE 算优势 -> 裁剪策略/价值损失更新；
    #   3. 调用 wrapper.create_trainer(...) 后 trainer.train()。
    # GAE 手算小例与 clip 行为验证见 tests/core/align_ppo_test.py。


if __name__ == "__main__":
    main()

"""手写 Trainer 的端到端 smoke 校验（CPU、合成数据）。

故意不 import llama31_8b_scratch.model / llama31_8b_scratch.data（并行开发中），而是用一个最小
``nn.Module`` + 合成 tensor 验证：
1. Trainer 能跑若干真实优化步并使 loss 下降；
2. 梯度累积生效（GAS=2 时每 2 次 forward 才 1 次 optimizer.step）；
3. checkpoint 能保存并恢复 global_step。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.training.trainer import Trainer, TrainerConfig


class ToyLinearModel(nn.Module):
    """一个能被 Trainer 直接训练的最小模块：线性回归 MSE。"""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.net = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 返回标量 loss（Trainer 支持 tensor 或 dict）
        return F.mse_loss(self.net(x), labels)


def _make_loader(n_batches: int = 20, mbs: int = 4, d: int = 8, seed: int = 0):
    """造一个固定目标 y = x @ W_true 的小数据集。"""
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(d, d, generator=g)
    loader = []
    for _ in range(n_batches):
        x = torch.randn(mbs, d, generator=g)
        y = x @ W_true
        loader.append({"x": x, "labels": y})
    return loader, W_true


class TestTrainerSmoke(unittest.TestCase):
    def test_loss_decreases_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainerConfig(
                output_dir=tmp,
                seed=0,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,  # GAS=2
                max_steps=20,
                learning_rate=0.1,
                weight_decay=0.0,
                warmup_steps=2,
                min_lr_ratio=0.0,
                max_grad_norm=1.0,
                mixed_precision="no",
                gradient_checkpointing=False,
                torch_compile=False,
                logging_steps=5,
                save_steps=10,
                eval_steps=10**9,
                save_total_limit=2,
                report_to="none",
            )
            loader, _ = _make_loader(n_batches=40, mbs=4, d=8)
            model = ToyLinearModel(d=8)

            trainer = Trainer(model=model, train_loader=loader, config=cfg,
                              device=torch.device("cpu"))
            # 记录初始 loss
            first_batch = loader[0]
            init_loss = float(model(first_batch["x"], first_batch["labels"]).item())

            trainer.train()

            final_loss = float(model(first_batch["x"], first_batch["labels"]).item())
            # 训练后在同 batch 上 loss 应明显下降
            self.assertLess(final_loss, init_loss * 0.8)
            # global_step 应走到 max_steps
            self.assertEqual(trainer.global_step, 20)

            # checkpoint-10 与 checkpoint-20 应存在
            ckpts = sorted(Path(tmp).glob("checkpoint-*"))
            self.assertGreaterEqual(len(ckpts), 1)

            # 续训：再建一个 Trainer 从 checkpoint-20 恢复，步数对齐
            model2 = ToyLinearModel(d=8)
            cfg2 = TrainerConfig(
                output_dir=tmp + "_resume",
                seed=0, per_device_train_batch_size=4,
                gradient_accumulation_steps=2, max_steps=25,
                learning_rate=0.1, warmup_steps=2, min_lr_ratio=0.0,
                logging_steps=5, save_steps=100, report_to="none",
                resume_from_checkpoint=str(Path(tmp) / "checkpoint-20"),
            )
            trainer2 = Trainer(model=model2, train_loader=loader, config=cfg2,
                                device=torch.device("cpu"))
            self.assertEqual(trainer2.global_step, 20)

    def test_gradient_accumulation_counts_optimizer_steps(self):
        """GAS=2、40 个 micro-batch、max_steps=5：应只调用 5 次 optimizer.step。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainerConfig(
                output_dir=tmp, seed=0,
                per_device_train_batch_size=4, gradient_accumulation_steps=2,
                max_steps=5, learning_rate=0.1, warmup_steps=0, min_lr_ratio=0.0,
                logging_steps=100, save_steps=100, report_to="none",
            )
            loader, _ = _make_loader(n_batches=20, mbs=4)
            model = ToyLinearModel()
            trainer = Trainer(model=model, train_loader=loader, config=cfg,
                              device=torch.device("cpu"))

            # 包一层计数器
            opt_step_count = {"n": 0}
            orig_step = trainer.optimizer.step
            def counted_step():
                opt_step_count["n"] += 1
                return orig_step()
            trainer.optimizer.step = counted_step  # type: ignore

            trainer.train()
            self.assertEqual(trainer.global_step, 5)
            self.assertEqual(opt_step_count["n"], 5)


if __name__ == "__main__":
    unittest.main()

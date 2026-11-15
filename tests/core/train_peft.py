"""peft_utils 的分支逻辑校验（不依赖 peft / bnb，环境里没装）。

校验：
1. ``count_trainable_parameters`` 在普通模型上正确统计；
2. 在 peft 缺失的环境下，``apply_lora`` 走 ``manual_freeze`` 分支，
   返回 backend=="manual_freeze"，且所有参数 requires_grad=False；
3. ``LoraConfig`` 默认字段合理；
4. ``build_bnb_quant_config`` 在 use_qlora=False 时返回 None。
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from llama31_8b_scratch.training.peft_utils import (
    LoraConfig,
    apply_lora,
    build_bnb_quant_config,
    count_trainable_parameters,
)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 8, bias=True)

    def forward(self, x):
        return self.head(self.v_proj(self.q_proj(x)))


class TestPeftUtils(unittest.TestCase):
    def test_count_trainable_parameters(self):
        model = DummyModel()
        stats = count_trainable_parameters(model)
        # 全部参数都 requires_grad=True
        self.assertEqual(stats["trainable"], stats["total"])
        self.assertGreater(stats["total"], 0)

    def test_apply_lora_manual_freeze_branch(self):
        """本环境未装 peft，应降级为把所有参数冻结。"""
        model = DummyModel()
        cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        wrapped, backend = apply_lora(model, cfg)

        # 在无 peft 环境下 backend 必然是 manual_freeze
        self.assertEqual(backend, "manual_freeze")
        # 所有参数都被冻结
        for p in wrapped.parameters():
            self.assertFalse(p.requires_grad)
        stats = count_trainable_parameters(wrapped)
        self.assertEqual(stats["trainable"], 0)

    def test_lora_config_defaults(self):
        cfg = LoraConfig()
        self.assertEqual(cfg.r, 8)
        self.assertEqual(cfg.lora_alpha, 16)
        self.assertIn("q_proj", cfg.target_modules)
        self.assertFalse(cfg.use_qlora)

    def test_bnb_config_none_when_disabled(self):
        cfg = LoraConfig(use_qlora=False)
        self.assertIsNone(build_bnb_quant_config(cfg))


if __name__ == "__main__":
    unittest.main()

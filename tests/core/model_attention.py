"""注意力实现与 GQA 测试（CPU）。

覆盖：eager / sdpa / flash_attention_2（缺失回退 sdpa）三种路径、
``repeat_kv`` 复制形状、GQA 前后张量形状。
"""

import unittest

import torch

from llama31_8b_scratch.model.attention import ScratchAttention, repeat_kv
from tests.core._scratch_test_utils import smoke_config


def _run_attention(impl: str, batch: int = 2, seq_len: int = 5):
    cfg = smoke_config()
    cfg.attention_impl = impl  # 测试注入注意力实现选择
    attn = ScratchAttention(cfg, layer_idx=0).eval()
    hidden = torch.randn(batch, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    return attn(hidden, position_ids=position_ids)


class TestRepeatKV(unittest.TestCase):
    def test_repeat_shape(self):
        # [B, n_kv=2, T, D=8]，每组 2 个 Q 头 -> [B, 4, T, 8]
        x = torch.randn(2, 2, 5, 8)
        out = repeat_kv(x, 2)
        self.assertEqual(out.shape, (2, 4, 5, 8))

    def test_repeat_group_one_identity(self):
        x = torch.randn(2, 4, 5, 8)
        self.assertTrue(torch.allclose(repeat_kv(x, 1), x))

    def test_repeat_values_aligned(self):
        x = torch.arange(2 * 2 * 1 * 4, dtype=torch.float32).reshape(2, 2, 1, 4)
        out = repeat_kv(x, 2)
        # 第 0、1 个 Q 头都复制自第 0 个 KV 头
        self.assertTrue(torch.allclose(out[:, 0], x[:, 0]))
        self.assertTrue(torch.allclose(out[:, 1], x[:, 0]))
        self.assertTrue(torch.allclose(out[:, 2], x[:, 1]))
        self.assertTrue(torch.allclose(out[:, 3], x[:, 1]))


class TestAttentionImpls(unittest.TestCase):
    def test_eager_output_shape_and_weights(self):
        out, weights, _ = _run_attention("eager")
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        # softmax 权重每行求和为 1
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(-1)), atol=1e-5))

    def test_sdpa_output_shape(self):
        out, weights, _ = _run_attention("sdpa")
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertIsNone(weights)

    def test_flash_falls_back_to_sdpa_when_missing(self):
        # 本机未安装 flash_attn，应触发回退 sdpa 且不抛异常
        out, _, _ = _run_attention("flash_attention_2")
        self.assertEqual(out.shape, (2, 5, 32))

    def test_eager_and_sdpa_close_numerically(self):
        torch.manual_seed(0)
        cfg = smoke_config()
        cfg.attention_impl = "eager"
        eager = ScratchAttention(cfg, layer_idx=0).eval()
        cfg.attention_impl = "sdpa"
        sdpa = ScratchAttention(cfg, layer_idx=0).eval()
        sdpa.load_state_dict(eager.state_dict())

        hidden = torch.randn(2, 5, 32)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        eager_out, _, _ = eager(hidden, position_ids=position_ids)
        sdpa_out, _, _ = sdpa(hidden, position_ids=position_ids)
        self.assertTrue(torch.allclose(eager_out, sdpa_out, rtol=1e-4, atol=1e-5))

    def test_gqa_forward_shapes(self):
        cfg = smoke_config(num_attention_heads=4, num_key_value_heads=2)
        attn = ScratchAttention(cfg, layer_idx=0).eval()
        hidden = torch.randn(2, 5, 32)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        out, _, _ = attn(hidden, position_ids=position_ids)
        self.assertEqual(out.shape, (2, 5, cfg.hidden_size))


if __name__ == "__main__":
    unittest.main()

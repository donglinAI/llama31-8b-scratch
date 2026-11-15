"""自回归生成：长度边界、eos 提前停止、贪心确定性与采样路径测试（CPU）。"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from llama31_8b_scratch.generation import generate, stream_generate
from tests.core._scratch_test_utils import build_causal_lm


class StubEosModel(nn.Module):
    """固定输出高 eos 对数几率的桩模型，用于验证提前停止分支。"""

    def __init__(self, eos_id: int, vocab_size: int = 64) -> None:
        super().__init__()
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        self.forward_calls += 1
        batch, _ = input_ids.shape
        logits = torch.full((batch, 1, self.vocab_size), -10.0)
        logits[:, -1, self.eos_id] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=None, loss=None)


class TestGenerateLengthAndStop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = build_causal_lm().eval()
        self.prompt = torch.randint(0, 64, (2, 4))

    def test_output_length_respects_bound(self):
        out = generate(self.model, self.prompt, max_new_tokens=4, greedy=True)
        self.assertLessEqual(out.shape[1], self.prompt.shape[1] + 4)
        self.assertGreaterEqual(out.shape[1], self.prompt.shape[1] + 1)

    def test_greedy_is_deterministic(self):
        first = generate(self.model, self.prompt, max_new_tokens=5, greedy=True)
        second = generate(self.model, self.prompt, max_new_tokens=5, greedy=True)
        self.assertTrue(torch.equal(first, second))

    def test_eos_stops_early(self):
        stub = StubEosModel(eos_id=2)
        prompt = torch.randint(0, 64, (1, 3))
        out = generate(stub, prompt, max_new_tokens=6, eos_token_id=2, use_cache=False)
        # 第一步即命中 eos，只追加 1 个 token
        self.assertEqual(out.shape[1], prompt.shape[1] + 1)
        self.assertEqual(stub.forward_calls, 1)

    def test_stream_generation_steps_grow(self):
        steps = list(
            stream_generate(self.model, self.prompt, max_new_tokens=3, greedy=True)
        )
        self.assertEqual(len(steps), 3)
        for index, state in enumerate(steps, start=1):
            self.assertEqual(state["sequence"].shape[1], self.prompt.shape[1] + index)
            self.assertEqual(state["token"].shape, (2, 1))
            self.assertFalse(state["finished"])

    def test_sampling_variants_run(self):
        # top-k / top-p / 重复惩罚 / 温度组合路径不报错且形状正确
        out = generate(
            self.model,
            self.prompt,
            max_new_tokens=3,
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.2,
            use_cache=True,
        )
        self.assertEqual(out.shape[0], 2)
        self.assertLessEqual(out.shape[1], self.prompt.shape[1] + 3)

    def test_no_cache_matches_cache_length(self):
        cached = generate(self.model, self.prompt, max_new_tokens=4, greedy=True)
        uncached = generate(
            self.model, self.prompt, max_new_tokens=4, greedy=True, use_cache=False
        )
        self.assertEqual(cached.shape, uncached.shape)


if __name__ == "__main__":
    unittest.main()

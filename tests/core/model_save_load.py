"""save_pretrained / load_weights 往返一致性与键名测试（CPU）。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from llama31_8b_scratch.model.causal_lm import ScratchForCausalLM
from tests.core._scratch_test_utils import build_causal_lm, smoke_config


class TestSaveLoadRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = build_causal_lm().eval()
        self.ids = torch.randint(0, 64, (2, 5))
        with torch.no_grad():
            self.reference_logits = self.model(self.ids).logits

    def test_artifacts_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            self.assertTrue((Path(tmp) / "model.safetensors").is_file())
            config_file = Path(tmp) / "config.json"
            self.assertTrue(config_file.is_file())
            config = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(config["model_type"], "llama")
            self.assertEqual(config["architectures"], ["LlamaForCausalLM"])

    def test_reloaded_weights_match_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)

            reloaded = ScratchForCausalLM(smoke_config()).eval()
            reloaded.load_weights(tmp)

            # 权重张量逐元素一致
            for key, value in self.model.state_dict().items():
                self.assertIn(key, reloaded.state_dict())
                self.assertTrue(
                    torch.allclose(value, reloaded.state_dict()[key], atol=1e-7),
                    f"权重不一致: {key}",
                )

            with torch.no_grad():
                reloaded_logits = reloaded(self.ids).logits
            self.assertTrue(
                torch.allclose(self.reference_logits, reloaded_logits, atol=1e-6)
            )

    def test_state_dict_keys_match_hf_llama(self):
        state = self.model.state_dict()
        expected_prefixes = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        for key in expected_prefixes:
            self.assertIn(key, state, f"缺少 HF 兼容键: {key}")


if __name__ == "__main__":
    unittest.main()

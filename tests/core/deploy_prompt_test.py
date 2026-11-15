"""prompt 渲染、工业后端参数拼装、gradio 历史转换测试（纯函数，无网络）。"""

import unittest

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.deployment.api_server import render_prompt
from llama31_8b_scratch.deployment.schemas import ChatMessage
from llama31_8b_scratch.deployment.vllm_serve import build_serve_kwargs as vllm_kwargs
from llama31_8b_scratch.deployment.vllm_serve import build_cli_command as vllm_cmd
from llama31_8b_scratch.deployment.sglang_serve import build_serve_kwargs as sglang_kwargs
from llama31_8b_scratch.deployment.sglang_serve import build_cli_command as sglang_cmd


class TestRenderPrompt(unittest.TestCase):
    def test_uses_llama_template(self):
        msgs = [
            ChatMessage(role="system", content="你是助手"),
            ChatMessage(role="user", content="你好"),
        ]
        prompt = render_prompt(msgs)
        # BOS 开头
        self.assertTrue(prompt.startswith("<|begin_of_text|>"))
        # 追加 assistant 角色头，等待模型续写
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)
        self.assertIn("你是助手", prompt)
        self.assertIn("你好", prompt)


class TestVllmConfig(unittest.TestCase):
    def test_kwargs_and_cli(self):
        cfg = Config.from_dict(
            {"vllm": {"model_path": "/m", "port": 8000, "max_model_len": 4096}}
        )
        kw = vllm_kwargs(cfg)
        self.assertEqual(kw["model"], "/m")
        self.assertEqual(kw["port"], 8000)
        self.assertTrue(kw["enable_prefix_caching"])
        cmd = vllm_cmd(cfg)
        self.assertIn("vllm.entrypoints.openai.api_server", cmd)
        self.assertIn("--enable-prefix-caching", cmd)


class TestSglangConfig(unittest.TestCase):
    def test_kwargs_and_cli(self):
        cfg = Config.from_dict({"sglang": {"model_path": "/s", "port": 8001}})
        kw = sglang_kwargs(cfg)
        self.assertEqual(kw["model_path"], "/s")
        self.assertTrue(kw["enable_radix_cache"])
        cmd = sglang_cmd(cfg)
        self.assertIn("sglang.launch_server", cmd)


class TestGradioHistory(unittest.TestCase):
    def test_tuple_history(self):
        # 延迟 import，避免在无 gradio 环境顶层依赖
        import sys
        sys.path.insert(
            0, "/home/user/Doubao/chats/38453210918078466/llama31-8b-scratch"
        )
        # 直接从 09_demo 脚本里取纯函数（脚本顶部有 shim，import 安全）
        import importlib.util
        path = (
            "/home/user/Doubao/chats/38453210918078466/llama31-8b-scratch/"
            "09_demo/gradio_app.py"
        )
        spec = importlib.util.spec_from_file_location("gradio_app_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        history = [["第一轮问", "第一轮答"], ["第二轮问", "第二轮答"]]
        msgs = mod.history_to_messages(history, "新问题")
        self.assertEqual(
            msgs,
            [
                {"role": "user", "content": "第一轮问"},
                {"role": "assistant", "content": "第一轮答"},
                {"role": "user", "content": "第二轮问"},
                {"role": "assistant", "content": "第二轮答"},
                {"role": "user", "content": "新问题"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

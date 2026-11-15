"""工业量化封装的惰性导入与分支逻辑测试（不真的装 auto_gptq/bnb/llama.cpp）。

按 SPEC 要求：
- NF4 概念不强行实测（环境里没有 bitsandbytes）；
- 验证缺库时抛出清晰 ImportError；
- 验证 GGUF 命令拼装正确、量化类型白名单生效。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llama31_8b_scratch.quantization import awq, bnb, gptq, gguf


class TestGptqLazy(unittest.TestCase):
    def test_missing_auto_gptq_raises_clear_error(self):
        with self.assertRaises(ImportError) as ctx:
            gptq.gptq_quantize("fake-model", "fake-out", ["hi"])
        self.assertIn("auto_gptq", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))


class TestAwqLazy(unittest.TestCase):
    def test_missing_autoawq_raises_clear_error(self):
        with self.assertRaises(ImportError) as ctx:
            awq.awq_quantize("fake-model", "fake-out", ["hi"])
        self.assertIn("autoawq", str(ctx.exception))


class TestBnbLazy(unittest.TestCase):
    """bitsandbytes 未安装时，load_quantized_model 应在某一步抛 ImportError。

    我们不强行实测 NF4 数值（SPEC 明确要求），只验证惰性分支的存在。
    """

    def test_build_config_does_not_require_bnb_at_import_time(self):
        # BitsAndBytesConfig 只是 transformers 里的 dataclass，不 import bnb；
        # 如果 transformers 也没装，则抛 ImportError；两种情况都算"惰性正常"。
        try:
            cfg4 = bnb.build_4bit_config()
            cfg8 = bnb.build_8bit_config()
        except ImportError as e:
            self.assertIn("transformers", str(e))
            self.skipTest("transformers 未安装，跳过 bnb config 构造")
        else:
            # 字段名是 transformers 约定，不绑死具体值
            self.assertTrue(hasattr(cfg4, "load_in_4bit"))
            self.assertTrue(hasattr(cfg8, "load_in_8bit"))

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            bnb.load_quantized_model("fake", mode="6bit")


class TestGgufCommandBuild(unittest.TestCase):
    def test_quant_type_whitelist(self):
        # 合法类型不抛
        for t in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"):
            self.assertIn(t, gguf.GGUF_QUANT_TYPES)
        # 非法类型抛 ValueError
        with self.assertRaises(ValueError):
            gguf.build_quantize_command(
                "in.gguf", "out.gguf", "Q99_XXX",
                llama_cpp_dir="/nonexistent",
            )

    def test_missing_convert_script_raises(self):
        with self.assertRaises(FileNotFoundError):
            gguf.build_convert_command(
                "hf-model", "out.gguf",
                llama_cpp_dir="/nonexistent-llama-cpp",
            )

    def test_pipeline_collects_both_commands(self):
        # 构造一个假的 llama.cpp 仓库目录，验证命令拼装逻辑
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")
            build = root / "build" / "bin"
            build.mkdir(parents=True)
            (build / "llama-quantize").write_text("# fake", encoding="utf-8")

            cmds = gguf.run_gguf_pipeline(
                "hf-model",
                str(root / "out"),
                llama_cpp_dir=str(root),
                quant_type="Q4_K_M",
                run=False,
            )
            self.assertEqual(cmds.convert[0], "python3")
            self.assertIn("convert_hf_to_gguf.py", cmds.convert[1])
            self.assertEqual(cmds.quantize[-1], "Q4_K_M")
            self.assertIn("llama-quantize", cmds.quantize[0])


if __name__ == "__main__":
    unittest.main()

"""配置加载器单元测试，仅依赖标准库与 PyYAML。"""

import tempfile
import unittest
from pathlib import Path

from llama31_8b_scratch.common.config import Config, ConfigError, deep_merge, load_config


class TestDeepMerge(unittest.TestCase):
    def test_nested_dict_merged(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}}
        merged = deep_merge(base, override)
        self.assertEqual(merged, {"a": {"x": 1, "y": 20, "z": 30}, "b": 3})

    def test_list_replaced_not_concatenated(self):
        merged = deep_merge({"a": [1, 2]}, {"a": [3]})
        self.assertEqual(merged["a"], [3])

    def test_original_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"y": 2}})
        self.assertEqual(base, {"a": {"x": 1}})


class TestConfig(unittest.TestCase):
    def test_attribute_and_item_access(self):
        cfg = Config.from_dict({"model": {"hidden_size": 4096}, "seed": 42})
        self.assertEqual(cfg.model.hidden_size, 4096)
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg.get("model.hidden_size"), 4096)
        self.assertIsNone(cfg.get("model.missing"))
        self.assertEqual(cfg.get("model.missing", 7), 7)

    def test_missing_attribute_raises(self):
        cfg = Config.from_dict({"a": 1})
        with self.assertRaises(AttributeError):
            _ = cfg.nope

    def test_override_parsing(self):
        cfg = Config.from_dict({"model": {"lr": 0.1, "layers": 2}})
        cfg.apply_overrides(["model.lr=0.5", "model.layers=4", "model.name=scratch"])
        self.assertEqual(cfg.model.lr, 0.5)
        self.assertEqual(cfg.model.layers, 4)
        self.assertEqual(cfg.model.name, "scratch")

    def test_bad_override(self):
        cfg = Config.from_dict({})
        with self.assertRaises(ConfigError):
            cfg.apply_overrides(["no_equal_sign"])

    def test_yaml_with_base_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "base.yaml").write_text(
                "model:\n  hidden_size: 128\n  layers: 2\nseed: 1\n",
                encoding="utf-8",
            )
            child = tmp / "child.yaml"
            child.write_text(
                "_base_: base.yaml\nmodel:\n  layers: 4\n",
                encoding="utf-8",
            )
            cfg = load_config(child)
            self.assertEqual(cfg.model.hidden_size, 128)
            self.assertEqual(cfg.model.layers, 4)
            self.assertEqual(cfg.seed, 1)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            Config.from_yaml("/path/that/never/exists.yaml")


if __name__ == "__main__":
    unittest.main()

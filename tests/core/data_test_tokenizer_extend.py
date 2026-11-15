"""词表扩充与 embedding 扩展测试（numpy，无 torch 依赖）。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from llama31_8b_scratch.data.process_pretrain import read_jsonl, write_jsonl
from llama31_8b_scratch.data.tokenizer_extend import expand_embedding, extend_vocab


class TestExtendVocab(unittest.TestCase):
    def test_dedup_and_mapping(self):
        vocab, mapping, added = extend_vocab(["a", "b"], ["b", "c", "d"])
        self.assertEqual(vocab, ["a", "b", "c", "d"])
        self.assertEqual(added, ["c", "d"])
        self.assertEqual(mapping["c"], 2)
        self.assertEqual(mapping["a"], 0)

    def test_no_duplicate_tokens(self):
        vocab, _, _ = extend_vocab(["x"], ["x", "x", "y"])
        self.assertEqual(len(vocab), 2)


class TestExpandEmbedding(unittest.TestCase):
    def test_shape_and_old_rows_kept(self):
        weight = np.random.randn(4, 8).astype(np.float32)
        expanded = expand_embedding(weight, num_new=3, initializer_range=0.02, seed=1)
        self.assertEqual(expanded.shape, (7, 8))
        np.testing.assert_allclose(expanded[:4], weight)
        # 新行接近 0（标准差 0.02，截断 ±0.04）
        self.assertLess(np.abs(expanded[4:]).max(), 0.05)

    def test_zero_new_returns_copy(self):
        weight = np.random.randn(4, 8).astype(np.float32)
        expanded = expand_embedding(weight, num_new=0)
        self.assertEqual(expanded.shape, weight.shape)
        self.assertFalse(np.shares_memory(expanded, weight))


class TestProcessOrchestrationIO(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        records = [{"text": "第一段足够长的中文样本内容。"}, {"text": "第二段足够长的中文样本内容。"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_jsonl(records, path)
            loaded = read_jsonl(path)
            self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()

"""tokenizers Rust BPE 后端真实训练与接口测试（CPU，toy 语料）。"""

import tempfile
import unittest
from pathlib import Path

from llama31_8b_scratch.common.paths import project_root
from llama31_8b_scratch.data.process_pretrain import read_jsonl
from llama31_8b_scratch.data.tokenizer_bpe import ScratchBPETokenizer


def _toy_texts() -> list[str]:
    records = read_jsonl(project_root() / "data/toy/pretrain.jsonl")
    return [r["text"] for r in records]


class TestScratchBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = ScratchBPETokenizer.train(
            texts=_toy_texts(),
            vocab_size=300,
            min_frequency=1,
            max_token_length=20,
            show_progress=False,
        )

    def test_vocab_size_positive(self):
        self.assertGreater(self.tokenizer.vocab_size(), 10)
        self.assertEqual(len(self.tokenizer), self.tokenizer.vocab_size())

    def test_special_tokens_registered(self):
        ids = self.tokenizer.special_token_ids()
        for token in [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "<|finetune_right_pad_token|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]:
            self.assertIn(token, ids)
            self.assertIsInstance(ids[token], int)

    def test_encode_decode_roundtrip_basic(self):
        text = "光合作用是植物的重要过程。"
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        # BOS 位于首位
        self.assertEqual(ids[0], self.tokenizer.bos_id())
        decoded = self.tokenizer.decode(ids[1:])
        self.assertIsInstance(decoded, str)
        self.assertTrue(len(decoded) > 0)

    def test_encode_messages_starts_with_bos(self):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "在的"},
        ]
        ids = self.tokenizer.encode_messages(messages)
        self.assertEqual(ids[0], self.tokenizer.bos_id())
        self.assertGreater(len(ids), 2)

    def test_token_id_query(self):
        bos = self.tokenizer.token_to_id("<|begin_of_text|>")
        self.assertEqual(self.tokenizer.id_to_token(bos), "<|begin_of_text|>")
        self.assertIsNone(self.tokenizer.token_to_id("<|not_exist|>"))

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tokenizer.save(tmp)
            reloaded = ScratchBPETokenizer.from_file(tmp)
            self.assertEqual(reloaded.vocab_size(), self.tokenizer.vocab_size())
            self.assertEqual(
                reloaded.token_to_id("<|eot_id|>"),
                self.tokenizer.token_to_id("<|eot_id|>"),
            )
            ids = reloaded.encode("测试文本", add_special_tokens=False)
            self.assertIsInstance(ids, list)

    def test_train_requires_texts(self):
        with self.assertRaises(ValueError):
            ScratchBPETokenizer.train()


if __name__ == "__main__":
    unittest.main()

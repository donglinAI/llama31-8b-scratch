"""数据集与 collator 形状、掩码测试（CPU，toy 语料，真实训练小 BPE）。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from llama31_8b_scratch.common.paths import project_root
from llama31_8b_scratch.data.collators import (
    PreferenceCollator,
    PretrainCollator,
    SFTCollator,
)
from llama31_8b_scratch.data.datasets import (
    GRPODataset,
    PreferenceDataset,
    PretrainDataset,
    SFTDataset,
)
from llama31_8b_scratch.data.process_pretrain import read_jsonl
from llama31_8b_scratch.data.tokenizer_bpe import ScratchBPETokenizer

ROOT = project_root()
TOY = ROOT / "data/toy"


def _build_tokenizer() -> ScratchBPETokenizer:
    texts: list[str] = []
    for record in read_jsonl(TOY / "pretrain.jsonl"):
        texts.append(record["text"])
    for record in read_jsonl(TOY / "sft.jsonl"):
        texts.append(" ".join(m["content"] for m in record["messages"]))
    return ScratchBPETokenizer.train(
        texts=texts, vocab_size=400, min_frequency=1, show_progress=False
    )


class DatasetFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = _build_tokenizer()
        cls.pad_id = cls.tokenizer.pad_id()


class TestPretrainDataset(DatasetFixture):
    def test_jsonl_dataset_shapes(self):
        ds = PretrainDataset(TOY / "pretrain.jsonl", self.tokenizer, max_length=64)
        self.assertEqual(len(ds), 12)
        item = ds[0]
        self.assertEqual(item["input_ids"].shape, item["labels"].shape)
        self.assertTrue(torch.equal(item["labels"], item["input_ids"]))
        self.assertEqual(item["attention_mask"].sum().item(), len(item["input_ids"]))

    def test_memmap_dataset(self):
        ids = np.arange(20, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "corpus.bin"
            ids.tofile(bin_path)
            ds = PretrainDataset(bin_path, self.tokenizer, max_length=4)
            self.assertEqual(len(ds), 5)
            item = ds[0]
            self.assertEqual(len(item["input_ids"]), 4)
            self.assertTrue(torch.equal(item["labels"], item["input_ids"]))


class TestSFTDataset(DatasetFixture):
    def setUp(self):
        self.ds = SFTDataset(TOY / "sft.jsonl", self.tokenizer, max_length=128)

    def test_length_and_alignment(self):
        self.assertEqual(len(self.ds), 10)
        for i in range(len(self.ds)):
            item = self.ds[i]
            self.assertEqual(item["input_ids"].shape, item["labels"].shape)

    def test_prompt_masked_in_labels(self):
        item = self.ds[0]
        labels = item["labels"].tolist()
        # 至少存在 prompt 位置的 -100，且末尾补全部分不是全 -100
        self.assertIn(-100, labels)
        self.assertTrue(any(x != -100 for x in labels))
        # 前缀为 -100，后段为真实 id
        first_real = next(idx for idx, x in enumerate(labels) if x != -100)
        self.assertTrue(all(x == -100 for x in labels[:first_real]))

    def test_collator_pads_labels_minus100(self):
        collator = SFTCollator(pad_token_id=self.pad_id)
        batch = collator([self.ds[0], self.ds[1]])
        max_len = batch["input_ids"].shape[1]
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["attention_mask"].shape[1], max_len)
        # pad 位置 label 为 -100
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())


class TestPreferenceDataset(DatasetFixture):
    def test_pair_shapes(self):
        ds = PreferenceDataset(TOY / "preference.jsonl", self.tokenizer, max_length=128)
        self.assertEqual(len(ds), 10)
        item = ds[0]
        self.assertIn("chosen_input_ids", item)
        self.assertIn("rejected_input_ids", item)
        self.assertEqual(item["chosen_attention_mask"].sum().item(),
                         len(item["chosen_input_ids"]))

    def test_collator_branches(self):
        ds = PreferenceDataset(TOY / "preference.jsonl", self.tokenizer, max_length=128)
        collator = PreferenceCollator(pad_token_id=self.pad_id)
        batch = collator([ds[0], ds[1]])
        for prefix in ("chosen", "rejected"):
            self.assertEqual(
                batch[f"{prefix}_input_ids"].shape,
                batch[f"{prefix}_labels"].shape,
            )
            self.assertEqual(
                batch[f"{prefix}_attention_mask"].shape[1],
                batch[f"{prefix}_input_ids"].shape[1],
            )


class TestGRPODataset(DatasetFixture):
    def test_prompt_and_answer(self):
        ds = GRPODataset(TOY / "grpo.jsonl", self.tokenizer, max_length=64)
        self.assertEqual(len(ds), 10)
        item = ds[0]
        self.assertIsInstance(item["answer"], str)
        self.assertGreater(len(item["prompt_input_ids"]), 0)
        self.assertEqual(
            item["prompt_attention_mask"].sum().item(),
            len(item["prompt_input_ids"]),
        )


class TestPretrainCollator(DatasetFixture):
    def test_padding_and_mask(self):
        ds = PretrainDataset(TOY / "pretrain.jsonl", self.tokenizer, max_length=64)
        collator = PretrainCollator(pad_token_id=self.pad_id)
        batch = collator([ds[0], ds[1], ds[2]])
        self.assertEqual(batch["input_ids"].shape[0], 3)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        # 真实 token 位置 mask=1，pad 位置 mask=0
        self.assertTrue((batch["attention_mask"][batch["input_ids"] != self.pad_id] == 1).all())
        # pad 位置 label 为 -100
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())


if __name__ == "__main__":
    unittest.main()

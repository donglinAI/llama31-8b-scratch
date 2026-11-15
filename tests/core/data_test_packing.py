"""序列打包纯逻辑测试（仅标准库）。"""

import unittest

from llama31_8b_scratch.data.packing import (
    concatenate_sequences,
    count_tokens,
    pack_blocks,
    pack_blocks_with_labels,
)


class TestPacking(unittest.TestCase):
    def test_concatenate_with_eos(self):
        seqs = [[1, 2], [3, 4, 5]]
        merged = concatenate_sequences(seqs, eos_id=99)
        self.assertEqual(merged, [1, 2, 99, 3, 4, 5, 99])

    def test_pack_blocks_fixed_length(self):
        ids = list(range(10))
        blocks = pack_blocks(ids, block_size=4, pad_id=0)
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertEqual(len(block), 4)
        self.assertEqual(blocks[0], [0, 1, 2, 3])
        self.assertEqual(blocks[2], [8, 9, 0, 0])

    def test_pack_blocks_exact_multiple(self):
        ids = list(range(8))
        blocks = pack_blocks(ids, block_size=4, pad_id=-1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1], [4, 5, 6, 7])

    def test_labels_aligned_with_ids(self):
        ids = [10, 20, 30, 40, 50]
        labels = [-100, -100, 30, 40, 50]
        id_blocks, label_blocks = pack_blocks_with_labels(
            ids, labels, block_size=3, pad_id=0, label_pad=-100
        )
        self.assertEqual(len(id_blocks), len(label_blocks))
        for idb, lab in zip(id_blocks, label_blocks):
            self.assertEqual(len(idb), len(lab))
        # 第二块 label 填充为 -100
        self.assertEqual(label_blocks[1], [40, 50, -100])

    def test_label_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pack_blocks_with_labels([1, 2], [1], block_size=2, pad_id=0)

    def test_count_tokens(self):
        self.assertEqual(count_tokens([[1, 2], [3], []]), 3)

    def test_invalid_block_size(self):
        with self.assertRaises(ValueError):
            pack_blocks([1, 2], block_size=0, pad_id=0)


if __name__ == "__main__":
    unittest.main()

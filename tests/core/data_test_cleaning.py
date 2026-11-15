"""清洗、去重与各任务数据校验纯逻辑测试（仅标准库）。"""

import unittest

from llama31_8b_scratch.data import cleaning_core
from llama31_8b_scratch.data.process_preference_core import clean_batch as clean_pref_batch
from llama31_8b_scratch.data.process_pretrain_core import clean_batch as clean_pretrain_batch
from llama31_8b_scratch.data.process_sft_core import clean_batch as clean_sft_batch


class TestNormalize(unittest.TestCase):
    def test_nfc_and_whitespace(self):
        text = "  你好，   世界。\n\n\n\n换行  "
        out = cleaning_core.normalize_text(text)
        self.assertEqual(out, "你好， 世界。\n\n换行")

    def test_control_chars_removed(self):
        text = "abc\x00def\x07"
        self.assertEqual(cleaning_core.normalize_text(text), "abcdef")

    def test_html_entities_unescaped(self):
        text = "a &amp; b <tag>c</tag>"
        out = cleaning_core.strip_html_tags(text)
        out = cleaning_core.normalize_text(out)
        self.assertIn("&", out)
        self.assertNotIn("<tag>", out)

    def test_hash_stable_across_whitespace(self):
        h1 = cleaning_core.text_hash("你好，  世界。")
        h2 = cleaning_core.text_hash("你好，世界。")
        self.assertEqual(h1, h2)

    def test_quality_filter_length(self):
        self.assertFalse(cleaning_core.is_quality_text("短", min_chars=5))
        self.assertTrue(cleaning_core.is_quality_text("这是一段足够长的正常中文文本。"))

    def test_dedupe_by_hash(self):
        records = [{"text": "第一段内容"}, {"text": "第一段内容"}, {"text": "第二段"}]
        unique = cleaning_core.deduplicate_by_hash(records)
        self.assertEqual(len(unique), 2)


class TestPretrainCore(unittest.TestCase):
    def test_clean_batch_keeps_valid_drops_bad(self):
        records = [
            {"text": "这是一段足够长的正常预训练文本内容。"},
            {"text": "短"},
            {"content": "缺 text 字段"},
            {"text": "这是一段足够长的正常预训练文本内容。"},
        ]
        cleaned, stats = clean_pretrain_batch(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(stats["raw"], 4)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["duplicates_removed"], 1)

    def test_url_stripped(self):
        cleaned, _ = clean_pretrain_batch(
            [{"text": "访问 https://example.com 查看更多足够长的内容描述。"}]
        )
        self.assertNotIn("http", cleaned[0]["text"])


class TestSFTCore(unittest.TestCase):
    def test_valid_messages(self):
        records = [
            {
                "messages": [
                    {"role": "system", "content": "设定"},
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "答案"},
                ]
            }
        ]
        cleaned = clean_sft_batch(records)
        self.assertEqual(len(cleaned), 1)

    def test_rejects_bad_role_order(self):
        records = [
            {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
        ]
        self.assertEqual(clean_sft_batch(records), [])

    def test_rejects_missing_assistant(self):
        records = [{"messages": [{"role": "user", "content": "只有用户"}]}]
        self.assertEqual(clean_sft_batch(records), [])

    def test_rejects_system_not_first(self):
        records = [
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "system", "content": "s"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ]
        self.assertEqual(clean_sft_batch(records), [])


class TestPreferenceCore(unittest.TestCase):
    def test_valid_pair(self):
        records = [
            {"prompt": "问题", "chosen": "更好的回答内容", "rejected": "更差的回答内容"}
        ]
        cleaned = clean_pref_batch(records)
        self.assertEqual(len(cleaned), 1)

    def test_rejects_equal_chosen_rejected(self):
        records = [{"prompt": "q", "chosen": "相同回答", "rejected": "相同回答"}]
        self.assertEqual(clean_pref_batch(records), [])

    def test_rejects_empty_field(self):
        records = [{"prompt": "", "chosen": "c", "rejected": "r"}]
        self.assertEqual(clean_pref_batch(records), [])


if __name__ == "__main__":
    unittest.main()

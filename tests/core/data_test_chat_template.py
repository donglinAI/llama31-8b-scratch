"""对话模板与特殊 token 常量测试（仅标准库，不依赖 torch）。"""

import unittest

from llama31_8b_scratch.data.chat_template import (
    BOS_TOKEN,
    EOT_TOKEN,
    END_HEADER_TOKEN,
    START_HEADER_TOKEN,
    LlamaChatTemplate,
    render_block,
    render_messages,
)


class TestLlamaChatTemplate(unittest.TestCase):
    def test_block_structure(self):
        block = render_block("user", "你好")
        self.assertEqual(
            block,
            f"{START_HEADER_TOKEN}user{END_HEADER_TOKEN}\n\n你好{EOT_TOKEN}",
        )

    def test_render_messages_matches_spec(self):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好"},
        ]
        rendered = render_messages(messages)
        expected = (
            f"{BOS_TOKEN}"
            f"{START_HEADER_TOKEN}system{END_HEADER_TOKEN}\n\n你是助手{EOT_TOKEN}"
            f"{START_HEADER_TOKEN}user{END_HEADER_TOKEN}\n\n你好{EOT_TOKEN}"
            f"{START_HEADER_TOKEN}assistant{END_HEADER_TOKEN}\n\n您好{EOT_TOKEN}"
        )
        self.assertEqual(rendered, expected)

    def test_no_system_block_omitted(self):
        rendered = render_messages(
            [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "在的"}]
        )
        self.assertNotIn("system", rendered.split(START_HEADER_TOKEN)[1])
        self.assertTrue(rendered.startswith(BOS_TOKEN))

    def test_apply_text_and_generation_prompt(self):
        tpl = LlamaChatTemplate()
        text_out = tpl.apply(text="你好")
        self.assertIn("user", text_out)
        gen_out = tpl.apply(
            messages=[{"role": "user", "content": "你好"}],
            add_generation_prompt=True,
        )
        self.assertTrue(gen_out.endswith(f"{START_HEADER_TOKEN}assistant{END_HEADER_TOKEN}\n\n"))

    def test_split_prompt_completion(self):
        messages = [
            {"role": "system", "content": "设定"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "答案"},
        ]
        prompt_text, completion = LlamaChatTemplate.split_prompt_completion(messages)
        self.assertEqual(completion, "答案")
        self.assertIn("问题", prompt_text)
        self.assertNotIn("答案", prompt_text)

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            render_block("moderator", "x")

    def test_invalid_message_raises(self):
        with self.assertRaises(ValueError):
            render_messages([{"role": "unknown", "content": "x"}])

    def test_constant_ids(self):
        # 官方 Llama3.1 固定 id，与 configs/model/llama3.1_8b.yaml 对齐
        self.assertEqual(LlamaChatTemplate.BOS_ID, 128000)
        self.assertEqual(LlamaChatTemplate.EOS_ID, 128001)
        self.assertEqual(LlamaChatTemplate.EOT_ID, 128009)
        self.assertEqual(LlamaChatTemplate.PAD_ID, 128004)
        self.assertEqual(LlamaChatTemplate.START_HEADER_ID, 128006)
        self.assertEqual(LlamaChatTemplate.END_HEADER_ID, 128007)


if __name__ == "__main__":
    unittest.main()

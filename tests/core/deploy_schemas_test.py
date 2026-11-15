"""deployment schema 校验与默认值测试。"""
import unittest
from pydantic import ValidationError
from llama31_8b_scratch.deployment.schemas import ChatCompletionRequest


class TestReqs(unittest.TestCase):
    def _req(self, **kw):
        base = {"model": "llama31-8b-scratch", "messages": [{"role": "user", "content": "hi"}]}
        base.update(kw)
        return ChatCompletionRequest(**base)

    def test_defaults(self):
        r = self._req()
        self.assertFalse(r.stream)
        self.assertEqual(r.temperature, 1.0)
        self.assertEqual(r.top_p, 1.0)
        self.assertIsNone(r.max_tokens)
        self.assertEqual(r.effective_max_tokens, 0)


if __name__ == "__main__":
    unittest.main()

    def test_messages_empty_rejected(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="m", messages=[])

    def test_temperature_range(self):
        with self.assertRaises(ValidationError):
            self._req(temperature=3.0)

    def test_top_p_range(self):
        with self.assertRaises(ValidationError):
            self._req(top_p=1.5)

    def test_alias_maps(self):
        r = self._req(max_completion_tokens=64)
        self.assertEqual(r.max_tokens, 64)
        r2 = self._req(max_tokens=32)
        self.assertEqual(r2.max_tokens, 32)

    def test_unknown_fields_ignored(self):
        r = self._req(extra_thing={"a": 1})
        self.assertEqual(r.model, "llama31-8b-scratch")

    def test_multiple_roles(self):
        r = self._req(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        )
        self.assertEqual(len(r.messages), 3)
        self.assertEqual(r.messages[0].role, "system")

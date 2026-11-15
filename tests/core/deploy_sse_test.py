"""SSE 帧格式与处理逻辑纯函数测试（无网络/端口/torch）。"""

import unittest

from llama31_8b_scratch.deployment import sse
from llama31_8b_scratch.deployment.schemas import (
    ChatCompletionRequest,
    StreamOptions,
)
from llama31_8b_scratch.deployment.service import (
    BaseModelService,
    GenerateParams,
    GenerationOutput,
    map_request_to_params,
)
from llama31_8b_scratch.deployment.api_server import (
    build_chat_response,
    build_models_payload,
    iter_chat_chunks,
    render_prompt,
)
from llama31_8b_scratch.deployment.client import (
    build_chat_request_body,
    build_request_headers,
)


class FakeService(BaseModelService):
    """内存假后端：不 import torch，按预定文本逐段产出。"""

    model_name = "fake"

    def generate(self, prompt: str, params: GenerateParams) -> GenerationOutput:
        return GenerationOutput(
            text="你好世界",
            prompt_tokens=len(prompt),
            completion_tokens=4,
            finish_reason="stop",
        )

    def generate_stream(self, prompt, params):
        for piece in ["你", "好", "世界"]:
            yield piece


class TestSSE(unittest.TestCase):
    def test_frame_format(self):
        self.assertEqual(sse.sse_frame("hello"), "data: hello\n\n")

    def test_done_sentinel(self):
        self.assertEqual(sse.sse_done(), "data: [DONE]\n\n")
        self.assertEqual(sse.DONE_SENTINEL, "[DONE]")

    def test_to_sse_json_single_line(self):
        frame = sse.to_sse_json({"a": 1, "中": "文"})
        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        # 单行 JSON，不含裸换行
        body = frame[len("data: "):-2]
        self.assertNotIn("\n", body)

    def test_media_type(self):
        self.assertIn("text/event-stream", sse.media_type())


class TestRequestMapping(unittest.TestCase):
    def _req(self, **kw):
        base = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
        base.update(kw)
        return ChatCompletionRequest(**base)

    def test_default_mapping(self):
        p = map_request_to_params(self._req(), default_max_tokens=128)
        self.assertIsInstance(p, GenerateParams)
        self.assertEqual(p.temperature, 1.0)
        self.assertEqual(p.top_p, 1.0)
        self.assertEqual(p.max_new_tokens, 128)
        self.assertEqual(p.repetition_penalty, 1.0)
        self.assertEqual(p.stop, [])

    def test_explicit_max_tokens(self):
        p = map_request_to_params(self._req(max_tokens=42))
        self.assertEqual(p.max_new_tokens, 42)

    def test_frequency_penalty_maps_to_repetition(self):
        p = map_request_to_params(self._req(frequency_penalty=0.5))
        self.assertAlmostEqual(p.repetition_penalty, 1.5)

    def test_stop_string_normalized(self):
        p = map_request_to_params(self._req(stop="STOP"))
        self.assertEqual(p.stop, ["STOP"])
        p2 = map_request_to_params(self._req(stop=["A", "B"]))
        self.assertEqual(p2.stop, ["A", "B"])


class TestResponseDict(unittest.TestCase):
    def test_non_stream_structure(self):
        req = ChatCompletionRequest(
            model="fake", messages=[{"role": "user", "content": "hi"}]
        )
        out = GenerationOutput(text="你好", prompt_tokens=5, completion_tokens=2)
        d = build_chat_response(req, out)
        self.assertEqual(d["object"], "chat.completion")
        self.assertEqual(d["model"], "fake")
        self.assertEqual(d["choices"][0]["message"]["content"], "你好")
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        self.assertEqual(d["usage"]["prompt_tokens"], 5)
        self.assertEqual(d["usage"]["completion_tokens"], 2)
        self.assertEqual(d["usage"]["total_tokens"], 7)

    def test_models_payload(self):
        d = build_models_payload(FakeService())
        self.assertEqual(d["object"], "list")
        self.assertEqual(d["data"][0]["id"], "fake")
        self.assertEqual(d["data"][0]["object"], "model")


class TestStreamChunks(unittest.TestCase):
    def test_chunk_sequence(self):
        req = ChatCompletionRequest(
            model="fake", messages=[{"role": "user", "content": "hi"}]
        )
        prompt = render_prompt(req.messages)
        chunks = list(iter_chat_chunks(req, prompt, FakeService()))
        # 首帧 + 3 内容帧 + 结束帧
        self.assertGreaterEqual(len(chunks), 5)
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        contents = [c["choices"][0]["delta"].get("content") for c in chunks]
        self.assertEqual("".join(x for x in contents if x), "你好世界")
        # 结束帧 finish_reason
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_usage_chunk_when_requested(self):
        req = ChatCompletionRequest(
            model="fake",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
        )
        chunks = list(iter_chat_chunks(req, "prompt", FakeService()))
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["choices"], [])
        self.assertGreater(usage_chunks[0]["usage"]["total_tokens"], 0)


class TestClientBody(unittest.TestCase):
    def test_body_fields(self):
        body = build_chat_request_body(
            model="fake",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=16,
            stream=True,
        )
        self.assertEqual(body["model"], "fake")
        self.assertEqual(body["temperature"], 0.5)
        self.assertTrue(body["stream"])
        self.assertEqual(body["max_tokens"], 16)

    def test_body_no_max_tokens_when_none(self):
        body = build_chat_request_body(model="fake", messages=[])
        self.assertNotIn("max_tokens", body)

    def test_headers(self):
        self.assertEqual(build_request_headers()["Accept"], "application/json")
        self.assertEqual(
            build_request_headers(stream=True)["Accept"], "text/event-stream"
        )


if __name__ == "__main__":
    unittest.main()

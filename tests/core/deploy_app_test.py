"""create_app 构建测试（不监听端口、不依赖网络/torch）。

回归用例：早期 chat_completions 返回注解写成 `StreamingResponse | dict`，
FastAPI 会把该联合类型当作 response_model 推导，而 StreamingResponse 不是
合法 pydantic 字段，导致应用根本无法构建。这里断言应用能成功装配且路由齐全。
"""

import unittest

from llama31_8b_scratch.deployment.api_server import create_app
from llama31_8b_scratch.deployment.service import (
    BaseModelService,
    GenerateParams,
    GenerationOutput,
)


class FakeService(BaseModelService):
    model_name = "llama31-8b-scratch-smoke"

    def generate(self, prompt: str, params: GenerateParams) -> GenerationOutput:
        return GenerationOutput(text="ok", prompt_tokens=1, completion_tokens=1)

    def generate_stream(self, prompt, params):
        yield "ok"

    def list_models(self):
        return ["llama31-8b-scratch-smoke"]


class TestCreateApp(unittest.TestCase):
    def test_app_builds_and_routes_exist(self):
        app = create_app(FakeService())
        paths = {r.path for r in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/v1/models", paths)
        self.assertIn("/v1/chat/completions", paths)

    def test_app_state_service_wired(self):
        app = create_app(FakeService())
        self.assertIsInstance(app.state.service, BaseModelService)


if __name__ == "__main__":
    unittest.main()

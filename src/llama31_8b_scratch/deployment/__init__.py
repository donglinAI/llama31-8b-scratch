"""部署子包：OpenAI 兼容服务、vLLM/SGLang 工业后端封装、网页 Demo 与容器化。

分层设计（与 SPEC 第 9 节"部署/ Demo"对齐）：

1. **协议层**（纯 pydantic / 纯函数，零重依赖，可直接单测）：
   - :mod:`llama31_8b_scratch.deployment.schemas`  OpenAI 风格请求/响应/流式 chunk 模型；
   - :mod:`llama31_8b_scratch.deployment.sse`      SSE 帧格式化纯函数。
2. **服务抽象层**（面向接口，可注入 fake）：
   - :mod:`llama31_8b_scratch.deployment.service`   `BaseModelService` 统一 generate 接口，
     并提供 Scratch 手写模型、vLLM 两套实现（重依赖全部惰性导入）。
3. **HTTP 层**（FastAPI 已安装，可直接 import；不真正起端口也能单测纯处理函数）：
   - :mod:`llama31_8b_scratch.deployment.api_server`  `/v1/chat/completions` 与 `/v1/models`；
   - :mod:`llama31_8b_scratch.deployment.client`     OpenAI SDK 示例 + 标准库 urllib 原生示例。
4. **工业后端封装**（惰性导入，缺失时给清晰提示）：
   - :mod:`llama31_8b_scratch.deployment.vllm_serve` / :mod:`llama31_8b_scratch.deployment.sglang_serve`。

模块顶层**只**依赖 pydantic、标准库与 fastapi（均为已安装依赖）；gradio / vllm /
sglang / openai / httpx / uvicorn / torch 一律在函数/方法内部 import，保证
``import llama31_8b_scratch.deployment`` 在最小环境下成功。
"""

from __future__ import annotations

from llama31_8b_scratch.deployment import schemas, sse  # noqa: F401

__all__ = ["schemas", "sse"]

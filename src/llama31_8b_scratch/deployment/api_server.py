"""FastAPI 服务：``/v1/chat/completions`` 与 ``/v1/models``，OpenAI 兼容。

设计要点
--------
- **路径与字段逐字对齐 OpenAI**：``POST /v1/chat/completions``、``GET /v1/models``；
  流式响应为 ``text/event-stream``，每帧 ``data: {json}\\n\\n``，末尾发
  ``data: [DONE]\\n\\n``（见 :mod:`llama31_8b_scratch.deployment.sse`）。
- **处理逻辑全部抽成纯函数**（:func:`render_prompt`、:func:`build_chat_response`、
  :func:`iter_chat_chunks`、:func:`build_models_payload`），不依赖端口、不依赖 TestClient，
  单测直接调用并断言 dict / 字符串；FastAPI 路由只是薄薄一层包装。
- **后端可注入**：应用通过 :func:`create_app(service)` 持有一个
  :class:`~llama31_8b_scratch.deployment.service.BaseModelService`；单测传 fake，真机传
  Scratch / vLLM 后端。

fastapi、starlette 为已安装依赖，可在模块顶层 import；本模块**不** import
uvicorn（那是入口脚本的事），也不 import 未安装的 httpx/openai/gradio。
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.deployment.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionChoice,
    ChatMessage,
    ModelInfo,
    UsageInfo,
)
from llama31_8b_scratch.deployment import sse
from llama31_8b_scratch.deployment.service import (
    BaseModelService,
    GenerationOutput,
    map_request_to_params,
)

logger = get_logger("llama31_8b_scratch.deployment.api_server")


# ---------------------------------------------------------------------------
# 纯函数：prompt 渲染
# ---------------------------------------------------------------------------


def render_prompt(messages: List[ChatMessage]) -> str:
    """把消息列表渲染成 Llama3.1 对话 prompt，并追加 assistant 角色头。

    复用数据代理冻结实现 :class:`llama31_8b_scratch.data.chat_template.LlamaChatTemplate`，
    不在此自拼模板字符串（SPEC 5.4）。
    """
    from llama31_8b_scratch.data.chat_template import LlamaChatTemplate  # noqa: PLC0415

    dicts = [m.model_dump() for m in messages]
    return LlamaChatTemplate().apply(messages=dicts, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 纯函数：非流式响应组装
# ---------------------------------------------------------------------------


def build_chat_response(
    request: ChatCompletionRequest,
    output: GenerationOutput,
) -> dict:
    """把一次生成结果组装成 OpenAI 非流式响应 dict。"""
    total = output.prompt_tokens + output.completion_tokens
    resp = ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant", content=output.text
                ),
                finish_reason=output.finish_reason or "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=total,
        ),
    )
    return resp.model_dump()


# ---------------------------------------------------------------------------
# 纯函数：流式 chunk 组装
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _chunk(
    model: str,
    request_id: str,
    created: int,
    delta: dict,
    finish_reason: str | None = None,
    usage: UsageInfo | None = None,
    choices: list | None = None,
) -> dict:
    """构造一个流式 chunk dict；``choices=None`` 时给默认的 index=0 候选。"""
    if choices is None:
        choices = [
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            ).model_dump()
        ]
    return ChatCompletionChunk(
        id=request_id,
        model=model,
        created=created,
        choices=choices,  # type: ignore[arg-type]
        usage=usage,
    ).model_dump()


def iter_chat_chunks(
    request: ChatCompletionRequest,
    prompt: str,
    service: BaseModelService,
) -> Iterator[dict]:
    """流式生成的纯迭代器：产出一个个 chunk dict（尚未包成 SSE 帧）。

    帧顺序严格对齐 OpenAI：

    1. 首帧：``delta={"role": "assistant"}``，无 content；
    2. 中间帧：每个文本片段一帧，``delta={"content": piece}``；
    3. 结束帧：``delta={}``，``finish_reason="stop"``；
    4. 若请求要求 usage：再发一帧 ``choices=[]``、带 ``usage`` 的统计帧。
    """
    params = map_request_to_params(request)
    request_id = _new_id()
    created = int(time.time())

    # 1) 首帧：角色声明
    yield _chunk(request.model, request_id, created, {"role": "assistant"})

    # 2) 内容增量
    prompt_tokens = service.count_tokens(prompt)
    completion_tokens = 0
    for piece in service.generate_stream(prompt, params):
        completion_tokens += service.count_tokens(piece)
        yield _chunk(request.model, request_id, created, {"content": piece})

    # 3) 结束帧
    yield _chunk(
        request.model, request_id, created, {}, finish_reason="stop"
    )

    # 4) 可选 usage 帧
    want_usage = bool(
        request.stream_options is not None and request.stream_options.include_usage
    )
    if want_usage:
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        yield _chunk(
            request.model,
            request_id,
            created,
            delta={},
            usage=usage,
            choices=[],
        )


# ---------------------------------------------------------------------------
# 纯函数：模型列表
# ---------------------------------------------------------------------------


def build_models_payload(service: BaseModelService) -> dict:
    """/v1/models 响应体。"""
    data = [ModelInfo(id=mid).model_dump() for mid in service.list_models()]
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# FastAPI 应用装配（薄包装，真正逻辑在上面纯函数里）
# ---------------------------------------------------------------------------


def create_app(service: BaseModelService) -> FastAPI:
    """构造 FastAPI 应用并注入模型服务。"""
    app = FastAPI(title="llama31-8b-scratch OpenAI-compatible API", version="0.1.0")
    app.state.service = service

    # 注意：所有路由都显式 response_model=None。
    # - 流式返回 StreamingResponse、非流式返回原始 dict；用联合类型注解（如
    #   `StreamingResponse | dict`）会被 FastAPI 当成 response_model 推导，
    #   而 StreamingResponse 不是合法 pydantic 字段类型，导致应用无法构建。
    # - response_model=None 让 FastAPI 跳过响应模型序列化，原样透传返回值。
    @app.get("/health", response_model=None)
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=None)
    def list_models() -> dict:
        return build_models_payload(service)

    @app.post("/v1/chat/completions", response_model=None)
    def chat_completions(request: ChatCompletionRequest):
        prompt = render_prompt(request.messages)
        params = map_request_to_params(request)

        if request.stream:
            def event_stream() -> Iterator[str]:
                for chunk in iter_chat_chunks(request, prompt, service):
                    yield sse.to_sse_json(chunk)
                yield sse.sse_done()

            return StreamingResponse(
                event_stream(),
                media_type=sse.media_type(),
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        output = service.generate(prompt, params)
        return build_chat_response(request, output)

    return app

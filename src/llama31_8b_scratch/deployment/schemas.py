
"""OpenAI 兼容的请求/响应 pydantic 模型。

目标：让本项目的服务在路径（``/v1/chat/completions``、``/v1/models``）、字段名、
SSE 帧格式、``[DONE]`` 结束标记上与 OpenAI 官方接口**逐字一致**，从而可以直接被
官方 ``openai`` SDK 或任意兼容客户端调用，无需改一行客户端代码。

本模块**只**依赖 pydantic 与标准库，不 import fastapi/torch，便于在 CPU、无网络
环境做 schema 校验单测。字段默认值对齐 OpenAI 官方语义：

- ``temperature`` 默认 1.0，范围 0~2；``top_p`` 默认 1.0，范围 0~1；
- ``stream`` 默认 ``False``；``max_tokens`` 为可选项（不传则用服务端默认上限）；
- ``messages`` 至少一条；``role`` 限定 system / user / assistant。
"""

from __future__ import annotations

import time
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 角色与 finish_reason 的字面量（与 OpenAI 语义一致）
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ChatMessage(BaseModel):
    """对话中的单条消息（请求入参与响应 message 共用结构）。"""

    role: Role
    content: str

    model_config = {"extra": "ignore"}


class StreamOptions(BaseModel):
    """流式附加选项，OpenAI 用它控制是否在最后一帧回传 usage。"""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` 的请求体。

    只覆盖本项目用到的字段；多余字段（如 ``tools``、``logprobs``）通过
    ``extra="ignore"`` 丢弃，保证客户端多传字段不报错。
    """

    model: str = Field(..., description="请求的模型名，需与 /v1/models 对应")
    messages: List[ChatMessage]
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大生成 token 数")
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[str | List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "ignore"}

    @field_validator("messages")
    @classmethod
    def _messages_non_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages 至少需要一条消息")
        return v

    @model_validator(mode="after")
    def _resolve_max_tokens(self) -> "ChatCompletionRequest":
        """OpenAI 新老两个 max 字段二选一；``max_completion_tokens`` 优先。"""
        if self.max_completion_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self

    @property
    def effective_max_tokens(self) -> int:
        """对外暴露的最大生成长度；未指定时由服务端给默认上限（见调用处）。"""
        return self.max_tokens if self.max_tokens is not None else 0


class UsageInfo(BaseModel):
    """token 用量统计，OpenAI 结构：prompt/completion/total 三项。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseMessage(BaseModel):
    """非流式响应里 assistant 产出的 message。"""

    role: str = "assistant"
    content: str = ""


class ChatCompletionChoice(BaseModel):
    """非流式响应的一个候选（本项目默认 batch=1，只有 index=0）。"""

    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Optional[FinishReason] = "stop"


class ChatCompletionResponse(BaseModel):
    """``POST /v1/chat/completions`` 非流式响应体。"""

    id: str = Field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# 流式 chunk
# ---------------------------------------------------------------------------


class ChatCompletionDelta(BaseModel):
    """流式增量：首帧给 ``role``，后续帧只给 ``content``，末帧两者皆空。"""

    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """流式响应的一个候选增量。"""

    index: int = 0
    delta: ChatCompletionDelta = Field(default_factory=ChatCompletionDelta)
    finish_reason: Optional[FinishReason] = None


class ChatCompletionChunk(BaseModel):
    """``data: `` 帧里承载的 JSON 对象，``object`` 固定为 ``chat.completion.chunk``。"""

    id: str = Field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice] = Field(default_factory=list)
    usage: Optional[UsageInfo] = None


class ModelInfo(BaseModel):
    """/v1/models 里单个模型条目。"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llama31-8b-scratch"


class ModelListResponse(BaseModel):
    """``GET /v1/models`` 响应体。"""

    object: str = "list"
    data: List[ModelInfo]

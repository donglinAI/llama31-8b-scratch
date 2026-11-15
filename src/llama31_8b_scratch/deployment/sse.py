"""SSE（Server-Sent Events）帧格式化纯函数。

OpenAI 流式接口本质是一个 ``text/event-stream`` 响应体，每一段增量都写成一帧：

.. code-block:: text

    data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n

最后再单独发一帧终止标记：

.. code-block:: text

    data: [DONE]\n\n

规范要点（本模块逐字对齐，便于单测做字符串断言）：

- 每行以 ``data: ``（**冒号后一个空格**）开头；
- 一帧可有多行 ``data:``，这里每个 chunk 只有一行；
- 帧与帧之间用**两个换行** ``\\n\\n`` 分隔；
- ``[DONE]`` 不是 JSON，是字面量，且之后不再发任何帧。

本模块只依赖标准库 ``json``，不 import fastapi / torch，可在任何环境单测。
"""

from __future__ import annotations

import json
from typing import Any, Dict

#: OpenAI 约定的结束标记字面量（不包成 JSON）
DONE_SENTINEL: str = "[DONE]"


def sse_frame(data: str) -> str:
    """把一段文本格式化成一帧 SSE ``data``。

    Args:
        data: 帧内容（应已完成 JSON 序列化或为 ``[DONE]`` 字面量）。

    Returns:
        ``"data: <data>\\n\\n"`` 形式的字符串。若传入的文本自身含换行，
        规范上每行都应加 ``data: `` 前缀；这里我们序列化的 JSON 从不换行，
        因此单行即可。
    """
    return f"data: {data}\n\n"


def to_sse_json(payload: Dict[str, Any]) -> str:
    """把一个响应 dict 序列化成单行 JSON，并包成一帧 SSE。

    使用 ``ensure_ascii=False`` 以保留中文原文；``separators`` 去掉多余空格，
    保证输出是单行、可逐字段比对的稳定字符串。
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return sse_frame(body)


def sse_done() -> str:
    """返回 OpenAI 约定的结束帧 ``"data: [DONE]\\n\\n"``。"""
    return sse_frame(DONE_SENTINEL)


def media_type() -> str:
    """SSE 响应应使用的 Content-Type。"""
    return "text/event-stream; charset=utf-8"


"""两种 OpenAI 兼容调用示例：官方 SDK 与标准库 urllib 原生 HTTP。

两种写法对着同一个 ``/v1/chat/completions``，目的：

1. **官方 SDK 版** :func:`chat_with_openai_sdk`：证明我们的服务真的兼容——装上
   ``openai`` 库后改个 ``base_url`` 就能用；``openai`` 未安装时惰性 import 并给提示；
2. **原生 urllib 版** :func:`chat_with_urllib`：不依赖任何第三方库，用标准库
   ``urllib.request`` 直接发 HTTP，便于在极简环境调试、也便于讲清请求体长什么样。

其中 :func:`build_chat_request_body` / :func:`build_request_headers` 是纯函数，
把请求体组装逻辑从网络调用里拆出来，可在无网络环境单测。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

#: 默认地址（与 run_api.py 一致）
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# 纯函数：请求体 / 请求头组装（可单测）
# ---------------------------------------------------------------------------


def build_chat_request_body(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """组装 ``/v1/chat/completions`` 的请求体 dict（纯函数）。"""
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    return body


def build_request_headers(stream: bool = False) -> Dict[str, str]:
    """组装请求头。SSE 流式时显式声明 ``Accept: text/event-stream``。"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    return headers


# ---------------------------------------------------------------------------
# 原生 urllib 调用
# ---------------------------------------------------------------------------


def chat_with_urllib(
    prompt: str,
    model: str = "llama31-8b-scratch",
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> str:
    """用标准库 urllib 发一次非流式对话，返回 assistant 文本。"""
    body = build_chat_request_body(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers=build_request_headers(stream=False),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def stream_chat_with_urllib(
    prompt: str,
    model: str = "llama31-8b-scratch",
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    max_tokens: int = 128,
):
    """用 urllib 读 SSE 流，逐段 yield assistant 文本（演示帧解析）。"""
    body = build_chat_request_body(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers=build_request_headers(stream=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece


# ---------------------------------------------------------------------------
# 官方 SDK 示例（惰性 import openai）
# ---------------------------------------------------------------------------


def stream_chat_messages(
    messages: List[Dict[str, str]],
    model: str = "llama31-8b-scratch",
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
):
    """通用流式调用：接受完整 messages 列表（多轮），逐段 yield assistant 文本。

    Gradio 多轮对话与测试都复用此函数；它只依赖标准库 urllib。
    """
    body = build_chat_request_body(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers=build_request_headers(stream=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece


def chat_with_openai_sdk(
    prompt: str,
    model: str = "llama31-8b-scratch",
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """用官方 ``openai`` 库调用（证明 OpenAI 兼容）。openai 未装时给提示。"""
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ImportError("请先 `pip install openai>=1.30` 再用 SDK 示例") from exc

    client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="not-needed")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""

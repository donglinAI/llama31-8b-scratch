"""Gradio 网页对话 Demo（惰性 import gradio）。

特性：
- 多轮对话：把 Gradio 的 (user, assistant) 历史转成 OpenAI messages；
- 流式输出：调用 API 的 ``stream:true``，逐字往界面追加；
- 参数调节：temperature / top_p / max_tokens 滑块。

它**不**直连模型，只通过标准库 urllib 访问 ``/v1/chat/completions``（见
:mod:`llama31_8b_scratch.deployment.client`），因此与手写后端 / vLLM / SGLang 完全解耦。

用法::

    python 09_demo/gradio_app.py --config configs/deploy/gradio.yaml
    # 前置：先在另一个终端启动 08_deploy/run_api.py

gradio 未安装时，``build_demo`` 才报错；本模块 import 阶段不 import gradio。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 三行 shim
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

from llama31_8b_scratch.common.config import load_config  # noqa: E402
from llama31_8b_scratch.common.logging_utils import get_logger, setup_logging  # noqa: E402
from llama31_8b_scratch.deployment.client import stream_chat_messages  # noqa: E402

logger = get_logger("gradio_demo")


def history_to_messages(
    history: List[Any], user_msg: str
) -> List[Dict[str, str]]:
    """把 Gradio 对话历史转成 OpenAI messages（纯函数，可单测）。

    兼容两种历史格式：
    - 旧格式 ``[(user, assistant), ...]``；
    - 新格式 ``[{"role":..., "content":...}, ...]``。
    末尾追加本轮 user 消息。
    """
    messages: List[Dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict) and "role" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            if u:
                messages.append({"role": "user", "content": str(u)})
            if a:
                messages.append({"role": "assistant", "content": str(a)})
    messages.append({"role": "user", "content": user_msg})
    return messages


def make_chat_fn(api_url: str, model: str):
    """返回一个供 Gradio ``ChatInterface`` / ``Chatbot`` 使用的流式回调。"""

    def respond(
        message: str,
        history: List[Any],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        messages = history_to_messages(history, message)
        buf = ""
        for piece in stream_chat_messages(
            messages,
            model=model,
            base_url=api_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
        ):
            buf += piece
            yield buf

    return respond


def build_demo(cfg):
    """构造 Gradio Blocks（此处才 import gradio）。"""
    import gradio as gr  # noqa: PLC0415

    g = cfg.gradio
    api_url = str(g.api_url)
    model = str(g.model)
    defaults = g.defaults
    respond = make_chat_fn(api_url, model)

    with gr.Blocks(title="llama31-8b-scratch Chat") as demo:
        gr.Markdown("# llama31-8b-scratch 网页对话\n后端：OpenAI 兼容 API")
        chatbot = gr.Chatbot(height=480)
        msg = gr.Textbox(placeholder="输入消息，Enter 发送")
        with gr.Row():
            temp = gr.Slider(0.0, 2.0, value=float(defaults.temperature), label="temperature")
            topp = gr.Slider(0.0, 1.0, value=float(defaults.top_p), label="top_p")
            maxt = gr.Slider(16, 2048, value=int(defaults.max_tokens), label="max_tokens")

        def user_submit(user_text, hist):
            return "", hist + [[user_text, None]]

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            lambda m, h, t, p, n: respond(m, h, t, p, n),
            [msg, chatbot, temp, topp, maxt],
            [chatbot],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="llama31-8b-scratch Gradio Demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)
    demo = build_demo(cfg)
    g = cfg.gradio
    demo.launch(
        server_name=str(getattr(g, "host", "0.0.0.0")),
        server_port=int(g.port),
        share=bool(getattr(g, "share", False)),
    )


if __name__ == "__main__":
    main()

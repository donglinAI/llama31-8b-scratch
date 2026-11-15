"""OpenAI 兼容客户端调用示例（同时演示 SDK 与纯 urllib 两种写法）。

用法（先启动 API）::

    python 08_deploy/run_api.py --config configs/deploy/api.yaml
    python 09_demo/openai_client_example.py

真正的实现都在 :mod:`llama31_8b_scratch.deployment.client`；本文件只是一个可直接
运行的演示脚本。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 三行 shim
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama31_8b_scratch.common.logging_utils import get_logger, setup_logging  # noqa: E402
from llama31_8b_scratch.deployment.client import (  # noqa: E402
    chat_with_urllib,
    stream_chat_with_urllib,
)

logger = get_logger("openai_client_example")


def main() -> None:
    setup_logging()
    prompt = "用一句话解释什么是 KV Cache。"

    # 1) 纯标准库 urllib，非流式
    logger.info("=== urllib 非流式 ===")
    reply = chat_with_urllib(prompt=prompt, max_tokens=128)
    logger.info("模型回复: %s", reply)

    # 2) 纯标准库 urllib，流式（逐字打印）
    logger.info("=== urllib 流式 ===")
    for piece in stream_chat_with_urllib(prompt=prompt, max_tokens=128):
        print(piece, end="", flush=True)
    print()

    # 3) 官方 openai SDK（需 pip install openai；未装时给出提示）
    logger.info("=== openai SDK ===")
    try:
        from llama31_8b_scratch.deployment.client import chat_with_openai_sdk

        sdk_reply = chat_with_openai_sdk(prompt=prompt)
        logger.info("SDK 回复: %s", sdk_reply)
    except ImportError as exc:
        logger.warning("跳过 SDK 示例: %s", exc)


if __name__ == "__main__":
    main()

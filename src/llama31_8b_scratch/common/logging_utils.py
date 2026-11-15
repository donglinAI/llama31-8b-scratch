"""统一日志工具。

仅封装标准库 logging，保证多进程（分布式训练）下格式一致；
不依赖 loguru 等第三方库，避免与 transformers/deepspeed 的日志体系冲突。
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO) -> None:
    """初始化根日志器，重复调用安全。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str = "llama31_8b_scratch", level: int | str | None = None) -> logging.Logger:
    """获取命名日志器，首次调用时自动初始化全局配置。"""
    if not _CONFIGURED:
        setup_logging(level or logging.INFO)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger

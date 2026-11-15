"""项目路径工具。

通过本文件位置反推项目根目录，避免在脚本里硬编码绝对路径，
保证仓库被克隆到任意位置都能正常工作。
"""

from __future__ import annotations

import os
from pathlib import Path


def src_dir() -> Path:
    """返回 ``src`` 目录的绝对路径。"""
    # 本文件位于 <root>/src/llama31_8b_scratch/common/paths.py
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """返回项目根目录的绝对路径。"""
    return src_dir().parent


def ensure_dir(path: str | os.PathLike) -> Path:
    """目录不存在则递归创建，返回 Path 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | os.PathLike) -> Path:
    """把相对路径解析为相对项目根目录的绝对路径，绝对路径原样返回。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()

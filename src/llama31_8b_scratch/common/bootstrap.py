"""零安装启动辅助。

当未执行 ``pip install -e .`` 时，编号目录下的入口脚本可借助本模块把
``src`` 目录注入 ``sys.path``。典型用法（入口脚本最顶部）::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from llama31_8b_scratch.common.config import load_config

注意：注入动作必须在 ``import llama31_8b_scratch`` 之前完成，因此入口脚本统一使用
上面的三行内联 shim；本模块额外提供 ``add_src_to_path`` 供测试与特殊场景使用。
"""

from __future__ import annotations

import sys
from pathlib import Path

from llama31_8b_scratch.common.paths import src_dir


def add_src_to_path() -> Path:
    """确保 ``src`` 目录在 ``sys.path`` 中，返回该目录路径。"""
    src = str(src_dir())
    if src not in sys.path:
        sys.path.insert(0, src)
    return Path(src)


def entry_shim(start_file: str | Path) -> None:
    """根据入口文件位置注入同级 ``src`` 目录。

    Args:
        start_file: 入口脚本的 ``__file__``，要求位于编号目录（根目录下一层）。
    """
    candidate = (Path(start_file).resolve().parents[1] / "src")
    src = str(candidate)
    if src not in sys.path:
        sys.path.insert(0, src)

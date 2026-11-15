"""随机种子工具，统一控制 Python / NumPy / PyTorch 的随机性。"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """设置全局随机种子。

    Args:
        seed: 种子值。
        deterministic: 为 True 时开启 CUDA 确定性算法，牺牲性能换取可复现。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 见 PyTorch 文档：use_deterministic_algorithms 会让无确定性实现的算子直接报错
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state() -> Optional[dict]:
    """导出当前各随机源状态，用于断点续训时精确恢复。"""
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

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
    os.environ["PYTHONHASHSEED"] = str(seed) # Python内置哈希（字典、set等）受该变量控制
    random.seed(seed) # Python标准库 random 的随机种子，对应 random.random() random.shuffle() 等原生随机函数
    np.random.seed(seed) # 设置 numpy 的随机种子，控制 np.random 系列随机采样

    import torch # 延迟导入 torch 

    torch.manual_seed(seed) # 控制CPU上 torch 的随机种子
    if torch.cuda.is_available(): # 设备是否有GPU；GPU是硬件，CUDA是调用GPU的软件接口，通过CUDA软件编程可调用硬件GPU
        torch.cuda.manual_seed_all(seed) # 设置所有GPU上的随机种子，单卡、多卡都覆盖；torch.cuda.manual_seed(42) 设置当前卡随机种子
    if deterministic: # cudnn是在CUDA里 卷积等计算算法，transformer架构少用，多在CNN作用较大
        torch.backends.cudnn.deterministic = True # 使用确定性算法
        torch.backends.cudnn.benchmark = False # 使用默认算法，不必测速选最快算法
        # 见 PyTorch 文档：use_deterministic_algorithms 会让无确定性实现的算子直接报错
        torch.use_deterministic_algorithms(True, warn_only=True) # 只使用确定性算法


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

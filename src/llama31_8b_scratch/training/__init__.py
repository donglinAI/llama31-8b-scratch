"""训练子包：预训练 + 指令微调的训练循环与工业级封装。

本包**顶层不 import 任何重型可选依赖**（peft / trl / deepspeed / accelerate /
bitsandbytes / wandb / tensorboard），保证仅安装 torch+numpy+yaml 时
``import llama31_8b_scratch.training`` 即可成功。重型依赖全部在函数内部惰性 import。
"""

from __future__ import annotations

from llama31_8b_scratch.training.losses import IGNORE_INDEX, causal_lm_loss, z_loss
from llama31_8b_scratch.training.optimizer import build_optimizer, build_param_groups
from llama31_8b_scratch.training.scheduler import CosineWarmupScheduler, build_scheduler
from llama31_8b_scratch.training.trainer import Trainer, TrainerConfig

__all__ = [
    "IGNORE_INDEX",
    "causal_lm_loss",
    "z_loss",
    "build_optimizer",
    "build_param_groups",
    "CosineWarmupScheduler",
    "build_scheduler",
    "Trainer",
    "TrainerConfig",
]

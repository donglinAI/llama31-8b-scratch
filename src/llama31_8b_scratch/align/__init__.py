"""对齐（Alignment）子包：RM / PPO / DPO / ORPO / GRPO 手写核心与工业封装。

顶层 import 仅依赖 ``torch`` / ``numpy`` / ``PyYAML``；``trl``、``peft``、
``accelerate`` 等重依赖只在 :mod:`llama31_8b_scratch.align.trl_wrappers` 的方法内部
惰性导入，保证核心依赖环境下 ``import llama31_8b_scratch.align`` 不报错。
"""

from llama31_8b_scratch.align.dpo import dpo_loss, implicit_reward
from llama31_8b_scratch.align.grpo import compute_grpo_advantages, grpo_loss
from llama31_8b_scratch.align.orpo import log_odds, orpo_loss, sequence_mean_logprobs
from llama31_8b_scratch.align.ppo import (
    ValueHead,
    clipped_policy_loss,
    clipped_value_loss,
    compute_gae,
    ppo_loss,
)
from llama31_8b_scratch.align.reward import pairwise_accuracy, pairwise_loss
from llama31_8b_scratch.align.utils import (
    gather_logprobs,
    masked_mean,
    normalize_advantages,
    sequence_logprobs,
    whiten,
)

__all__ = [
    # utils
    "gather_logprobs",
    "sequence_logprobs",
    "masked_mean",
    "normalize_advantages",
    "whiten",
    # reward
    "pairwise_loss",
    "pairwise_accuracy",
    # ppo
    "ValueHead",
    "compute_gae",
    "clipped_policy_loss",
    "clipped_value_loss",
    "ppo_loss",
    # dpo
    "dpo_loss",
    "implicit_reward",
    # orpo
    "sequence_mean_logprobs",
    "log_odds",
    "orpo_loss",
    # grpo
    "compute_grpo_advantages",
    "grpo_loss",
]

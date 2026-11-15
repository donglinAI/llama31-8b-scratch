"""优化器构造：AdamW（标准）与 AdamW8bit（bitsandbytes，惰性导入）。

为什么用 AdamW 而不是 Adam：
- Adam 的权重衰减是直接乘在梯度上的（``w * lr * wd``），与自适应学习率耦合；
- AdamW（Loshchilov & Hutter, 2019）把权重衰减从梯度更新里解耦，单独在更新前对
  参数做 ``w <- w - lr * wd * w``，这是大模型预训练的事实标准（GPT-3 / Llama 全系）。

为什么需要 AdamW8bit：
- 标准 AdamW 为每个参数维护两份动量状态（m、v），都用 fp32 存，大小是模型参数的 2 倍。
  7B 模型：参数 ~14GB（fp16），优化器状态 ~56GB（fp32 m+v），单卡 80GB 刚够；
- 8-bit Adam（Dettmers et al., 2022, bitsandbytes）把 m、v 量化到 8 bit，优化器状态
  显存从 2 倍 fp32 降到 0.5 倍 fp32，省 75% 优化器显存，且精度损失在大模型上可忽略；
- 关键：bitsandbytes 只能在有 GPU 的 Linux 上 import，所以这里必须**惰性导入**，
  CPU / 无 bnb 环境走标准 AdamW 并给出清晰提示。

关于参数分组（param groups）：
- 一般对所有权重矩阵做权重衰减，对 bias 与 RMSNorm 权重不做衰减；
- LoRA 场景下可选让 LoRA 参数用更大的 lr、冻结的 base 参数不更新。
  本模块提供 ``build_param_groups`` 自动按"2 维及以上张量"识别权重矩阵。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)

__all__ = ["build_param_groups", "build_optimizer"]


def build_param_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    no_decay_keywords: Iterable[str] = ("bias", "norm", "RMSNorm", "layernorm"),
    lr: float = 3e-4,
) -> List[Dict[str, Any]]:
    """把模型参数分成"衰减"与"不衰减"两组，供 AdamW 使用。

    规则：
    - 参数名命中 ``no_decay_keywords``（bias、各类 norm 权重）→ 不做权重衰减；
    - 其余参数（通常是 2D 及以上的权重矩阵）→ 做 ``weight_decay`` 衰减。

    这与 HF ``Trainer`` 的默认分组逻辑一致，是大模型训练的惯例：norm 与 bias 的尺度
    本身就是学出来的，再乘一个衰减会拖慢收敛。

    Args:
        model: 任意 ``nn.Module``。
        weight_decay: 权重衰减系数，预训练常用 0.1。
        no_decay_keywords: 命中即不衰减的子串（小写匹配）。
        lr: 这一组参数的学习率；多分组时可在外部覆写。

    Returns:
        可直接传给 ``AdamW(params=groups)`` 的列表。
    """
    no_decay_names = set()
    decay_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        hit = any(kw.lower() in name.lower() for kw in no_decay_keywords)
        (no_decay_names if hit else decay_names).add(name)

    # 用 id() 去重：同名参数不应出现两次
    no_decay_params = [
        p for n, p in model.named_parameters() if n in no_decay_names and p.requires_grad
    ]
    decay_params = [
        p for n, p in model.named_parameters() if n in decay_names and p.requires_grad
    ]

    groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ]
    # 过滤空组，避免 AdamW 报警告
    return [g for g in groups if len(g["params"]) > 0]


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
    no_decay_keywords: Iterable[str] = ("bias", "norm", "RMSNorm", "layernorm"),
    param_groups: Optional[List[Dict[str, Any]]] = None,
) -> torch.optim.Optimizer:
    """根据配置构造优化器。

    Args:
        model: 待优化模型。
        lr: 峰值学习率（会被 scheduler 进一步缩放）。
        weight_decay: 权重衰减。
        betas: Adam 的 (beta1, beta2)；大模型惯例 (0.9, 0.95)，区别于 CV 的 (0.9, 0.999)。
        eps: Adam 的 epsilon，默认 1e-8。
        optimizer_type: ``"adamw"`` 或 ``"adamw8bit"``；后者惰性导入 bitsandbytes。
        no_decay_keywords: 不做衰减的参数名子串。
        param_groups: 外部已分好的组（例如 LoRA 不同 lr）；为 None 时自动分组。

    Returns:
        构造好的优化器实例。

    Note:
        ``adamw8bit`` 在 bitsandbytes 缺失 / 无 GPU 时**静默回退**到标准 AdamW 并打
        warning，保证 CPU smoke 测试与单卡 CPU 调试不崩。真要用 8bit 节省显存，请在
        CUDA 环境安装 ``bitsandbytes``。
    """
    if param_groups is None:
        param_groups = build_param_groups(
            model,
            weight_decay=weight_decay,
            no_decay_keywords=no_decay_keywords,
            lr=lr,
        )

    opt_type = optimizer_type.lower()
    if opt_type == "adamw":
        return AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    if opt_type in ("adamw8bit", "adamw_bnb_8bit", "paged_adamw_8bit"):
        # 惰性导入：bitsandbytes 只在 CUDA Linux 可用，不污染 CPU / mac 环境
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as exc:  # pragma: no cover - 取决于环境
            logger.warning(
                "bitsandbytes 不可用（%s），回退到标准 AdamW；如需 8bit 优化器请在 "
                "CUDA Linux 环境 pip install bitsandbytes",
                exc,
            )
            return AdamW(
                param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )

        # paged 版本会把优化器状态 offload 到 CPU RAM，进一步省显存
        if opt_type == "paged_adamw_8bit":
            optim_cls = bnb.optim.PagedAdamW8bit
        else:
            optim_cls = bnb.optim.AdamW8bit
        return optim_cls(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"未知 optimizer_type: {optimizer_type}")

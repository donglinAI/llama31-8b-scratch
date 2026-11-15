"""分布式训练封装：DeepSpeed ZeRO 1/2/3 与 PyTorch FSDP。

本模块**全部惰性 import** deepspeed / torch.distributed，CPU 单测与 smoke 环境绝不
触发真正的多进程初始化。它的作用是：
1. 讲清 ZeRO 三个 stage 各切了什么、显存怎么省；
2. 讲清 FSDP 与 ZeRO 的 tradeoff；
3. 给入口脚本一组可调用的包装函数，真机上 ``torchrun`` 起来后直接用。

一、ZeRO（Zero Redundancy Optimizer）切了什么

数据并行（DDP）每张卡都持有完整的：模型权重 + 梯度 + 优化器状态（m、v）。
4 卡训 7B 时，每张卡显存里都有这三份冗余。ZeRO 按这三份冗余逐级切分：

+----------+----------------------+---------------------------+-----------------------+
| Stage    | 切优化器状态         | 切梯度                    | 切模型权重            |
+==========+======================+===========================+=======================+
| ZeRO-1   | ✅ 按参数分片         | ❌ 每张卡留完整梯度        | ❌ 每张卡留完整权重    |
+----------+----------------------+---------------------------+-----------------------+
| ZeRO-2   | ✅                   | ✅ 按参数分片              | ❌                    |
+----------+----------------------+---------------------------+-----------------------+
| ZeRO-3   | ✅                   | ✅                        | ✅ 按参数分片          |
+----------+----------------------+---------------------------+-----------------------+

- **ZeRO-1**：把 Adam 的 m、v 按卡数切分。优化器状态省 1/N，通信量与 DDP 相同。
  7B 单卡训练的最大瓶颈往往就是优化器状态（fp32 m+v ≈ 2×参数量），所以 ZeRO-1 性价比最高。
- **ZeRO-2**：进一步把梯度也切分。反向传播时每张卡只算自己那片梯度的本地范数，
  再 all-gather 归约，省一份梯度显存。通信量略增。
- **ZeRO-3**：把模型权重也切了。前向 / 反向时按需 all-gather 对应层的权重，用完释放。
  可以单卡训 70B，但通信量最大、实现最复杂；通常要配合 ``offload_optimizer`` /
  ``offload_param`` 把部分状态卸到 CPU/NVMe。

二、FSDP（Fully Sharded Data Parallel）是什么、与 ZeRO-3 的关系

PyTorch FSDP 是 PyTorch 官方的"ZeRO-3 等价物"：
- 同样把权重 / 梯度 / 优化器状态按卡数切分；
- 区别在于 FSDP 是**按 layer 粒度** all-gather（forward 时拉这一层、backward 后立即释放），
  与 ZeRO-3 思路几乎一样；
- tradeoff：
  - DeepSpeed ZeRO 生态成熟、配置式、与 HuggingFace 集成最好；
  - FSDP 是 PyTorch 原生、更轻量、与 ``torch.compile`` / ``torch.checkpoint`` 配合更紧，
    新版 PyTorch（2.x）FSDP2 在性能上已追平甚至超过 ZeRO-3。
- 8B 级训练：ZeRO-2 通常够用；70B+ 或单卡显存吃紧时再上 ZeRO-3 / FSDP + offload。

三、启动方式

真机多卡统一用 ``torchrun``（或 ``deepspeed`` 命令）启动：

.. code-block:: bash

    torchrun --nproc_per_node=8 run_pretrain.py --config configs/pretrain/8b.yaml \
        --override training.distributed=deepspeed \
        --override training.deepspeed_config=configs/pretrain/ds_z2.json
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedInfo",
    "get_dist_info",
    "init_distributed",
    "wrap_deepspeed",
    "wrap_fsdp",
    "wrap_ddp",
]


class DistributedInfo:
    """打包当前进程的分布式身份信息。"""

    def __init__(self) -> None:
        self.world_size: int = int(os.environ.get("WORLD_SIZE", 1))
        self.rank: int = int(os.environ.get("RANK", 0))
        self.local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
        self.enabled: bool = self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DistributedInfo(world_size={self.world_size}, rank={self.rank}, "
            f"local_rank={self.local_rank})"
        )


def get_dist_info() -> DistributedInfo:
    """从环境变量读取分布式身份（torchrun 启动时会注入 WORLD_SIZE/RANK/LOCAL_RANK）。"""
    return DistributedInfo()


def init_distributed(backend: str = "nccl") -> DistributedInfo:
    """初始化 ``torch.distributed``。单卡 / 无多卡环境直接返回，不报错。

    真正调用 ``dist.init_process_group`` 只在 ``world_size > 1`` 时发生。
    """
    info = get_dist_info()
    if not info.enabled:
        logger.info("未检测到多卡环境，跳过 dist.init_process_group")
        return info

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed 不可用，但 WORLD_SIZE>1")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(info.local_rank)
    logger.info("torch.distributed 已初始化：%s", info)
    return info


def wrap_ddp(
    model: nn.Module,
    info: Optional[DistributedInfo] = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """普通 DDP 包装（ZeRO-0 等价，每张卡完整复制模型）。"""
    info = info or get_dist_info()
    if not info.enabled:
        return model
    device_ids = [info.local_rank] if torch.cuda.is_available() else None
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_deepspeed(
    model: nn.Module,
    optimizer: Optional[Any] = None,
    training_config: Optional[Dict[str, Any]] = None,
    deepspeed_config: Optional[Dict[str, Any]] = None,
) -> tuple[nn.Module, Any, Any, int]:
    """用 DeepSpeed 包装模型 / 优化器（惰性 import）。

    Args:
        model: 待包装模型。
        optimizer: 已构造好的优化器；若为 None，DeepSpeed 会从 config 里自己建。
        training_config: 训练超参 dict（含 train_batch_size 等）。
        deepspeed_config: ds_z1/z2/z3.json 解析后的 dict。

    Returns:
        ``(engine, optimizer, lr_scheduler, global_rank)``。

    Note:
        CPU / 无 deepspeed 时直接把原模型与优化器原样返回，不抛异常，方便单测 dry import。
    """
    try:
        import deepspeed  # type: ignore
    except Exception as exc:
        logger.warning("deepspeed 不可用（%s），原样返回 model/optimizer", exc)
        return model, optimizer, None, 0

    engine, opt, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=list(model.parameters()),
        optimizer=optimizer,
        config=deepspeed_config,
        config_params=training_config,
    )
    info = get_dist_info()
    return engine, opt, getattr(engine, "lr_scheduler", None), info.rank


def wrap_fsdp(
    model: nn.Module,
    info: Optional[DistributedInfo] = None,
    mixed_precision: Optional[Any] = None,
    use_orig_params: bool = True,
) -> nn.Module:
    """用 PyTorch FSDP 包装（ZeRO-3 等价、PyTorch 原生）。

    Args:
        model: 待包装模型。
        info: 分布式身份；单卡时直接返回。
        mixed_precision: 可选 ``torch.distributed.fsdp.MixedPrecision``。
        use_orig_params: 为 True 时保留原参数名，便于 param groups / LoRA。
    """
    info = info or get_dist_info()
    if not info.enabled:
        return model

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision as FSDPMixedPrecision
    from torch.distributed.fsdp import ShardingStrategy

    if mixed_precision is None and torch.cuda.is_available():
        mixed_precision = FSDPMixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # FULL_SHARD = ZeRO-3；SHARD_GRAD_OP = ZeRO-2；NO_SHARD = DDP
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=info.local_rank,
        use_orig_params=use_orig_params,
    )

"""手写训练循环：预训练 / SFT 共用的工业级 Trainer。

本模块不依赖 HF Trainer / Accelerate，所有关键机制自己实现，便于在视频里逐行讲清：
梯度累积、梯度裁剪、混合精度 autocast、梯度检查点、checkpoint 存/续、torch.compile、
wandb/tensorboard 惰性可切换日志。

一、全局 batch size 的拆解（必须先建立直觉）

设：
- ``MBS`` = 每张卡每次 forward 的 micro-batch 大小（``per_device_train_batch_size``）；
- ``GAS`` = 梯度累积步数（``gradient_accumulation_steps``）；
- ``N`` = 参与训练的 GPU 数（world_size）。

则一次**优化器更新**看到的 token 总量（全局 batch size）为：

.. math::
    GLOBAL\\_BS = MBS \\times GAS \\times N

为什么要拆成 MBS × GAS × N 而不是直接一个大数？
- 单卡显存装不下大 batch（序列长 + 模型大），所以用 **梯度累积** 把 N 次小 forward
  的梯度加起来，等效于一次大 batch 的梯度；
- 多卡时每张卡独立算自己那一份 micro-batch，DDP/ZeRO 再把梯度 all-reduce 聚合，
  等效于更大的 batch。

调度器的单位是"优化步"，不是 forward 步：每 GAS 次 forward、每张卡各 forward 一次，
才 ``optimizer.step()`` 一次，``scheduler.step()`` 一次。

二、混合精度（autocast）

- 训练时权重保持 fp32（ master weights ），forward / backward 在 bf16（推荐，A100+）
  或 fp16 下进行；优化器更新仍在 fp32 上做。这就是"混合精度"。
- CPU 上 ``torch.autocast`` 对 bf16/fp16 支持有限，smoke 测试显式走 fp32 路径；
  真机配置（8B）才写 bf16。

三、梯度检查点（gradient checkpointing）

Transformer 自注意力 + MLP 的激活显存是 ``O(B * T * hidden * layers)``。梯度检查点
在前向时**不保存**中间激活，反向时重新算一遍，用 30% 额外算力换 ~60% 激活显存。
我们优先调用模型上的 ``gradient_checkpointing_enable()``（ScratchForCausalLM 已实现）；
模型没实现时给出 warning，不强行手动包。

四、日志后端（惰性、可切换）

``report_to`` 支持 ``none`` / ``wandb`` / ``tensorboard`` / ``all``。wandb 与 tensorboard
都在**首次写日志时**才 import，缺库时静默回退到纯控制台，绝不让 import 失败拖垮训练。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.common.paths import ensure_dir
from llama31_8b_scratch.common.seed import set_seed
from llama31_8b_scratch.training.optimizer import build_optimizer
from llama31_8b_scratch.training.scheduler import CosineWarmupScheduler, build_scheduler

logger = get_logger(__name__)

__all__ = ["Trainer", "TrainerConfig"]


# ----------------------------------------------------------------------
# 配置（用 dataclass 显式列出字段，避免到处 cfg.get("...", default)）
# ----------------------------------------------------------------------
class TrainerConfig:
    """训练超参的显式容器，字段与 YAML 节点一一对应。

    从 ``Config`` 对象构造：``TrainerConfig.from_config(cfg.training)``。
    """

    def __init__(
        self,
        output_dir: str,
        seed: int = 42,
        # batch 语义
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_steps: int = -1,
        num_train_epochs: float = 1.0,
        # 优化
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        adam_betas: tuple = (0.9, 0.95),
        adam_eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        optimizer_type: str = "adamw",
        # schedule
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
        # 精度 / 性能
        mixed_precision: str = "no",  # no / fp16 / bf16
        gradient_checkpointing: bool = False,
        torch_compile: bool = False,
        # 日志 / 存点
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: int = 3,
        report_to: str = "none",  # none / wandb / tensorboard / all
        project_name: str = "llama31-8b-scratch",
        run_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        # 数据
        dataloader_num_workers: int = 0,
        **_: Any,  # 吸收 YAML 里多余字段，向前兼容
    ) -> None:
        # YAML 1.1 会把无引号的 no/yes 解析成布尔；这里统一归一，避免 .lower() 崩溃
        if not isinstance(mixed_precision, str):
            mixed_precision = "no"
        if not isinstance(report_to, str):
            report_to = "none"

        self.output_dir = output_dir
        self.seed = seed
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_betas = tuple(adam_betas)
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm
        self.optimizer_type = optimizer_type
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.torch_compile = torch_compile
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        self.report_to = report_to
        self.project_name = project_name
        self.run_name = run_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.dataloader_num_workers = dataloader_num_workers

    @classmethod
    def from_config(cls, cfg: Config) -> "TrainerConfig":
        """从 ``cfg.training`` 节点构造。``cfg`` 可以是 Config 或 dict。"""
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        return cls(**data)


# ----------------------------------------------------------------------
# 日志后端（惰性）
# ----------------------------------------------------------------------
class _LogWriter:
    """统一的日志写入器：控制台 + 可选 wandb / tensorboard。

    wandb / tensorboard 在 ``__init__`` 时按 ``report_to`` 决定是否启用；真正的
    ``import`` 延后到首次 ``log()``，缺库时打一次 warning 后静默跳过。
    """

    def __init__(self, report_to: str, project: str, run_name: str, output_dir: str) -> None:
        self.report_to = (report_to or "none").lower()
        self._wandb = None
        self._tb_writer = None
        self._tb_tried = False

        want_wandb = self.report_to in ("wandb", "all")
        want_tb = self.report_to in ("tensorboard", "all")

        if want_wandb:
            try:
                import wandb  # type: ignore

                self._wandb = wandb.init(
                    project=project, name=run_name, dir=output_dir,
                    config={"report_to": report_to},
                )
                logger.info("wandb 已启动：project=%s run=%s", project, run_name)
            except Exception as exc:
                logger.warning("wandb 初始化失败（%s），回退到控制台", exc)
                self._wandb = None

        if want_tb:
            # tensorboard 也惰性 import
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
                self._tb_tried = True
                logger.info("tensorboard 已启动：%s", self._tb_writer.log_dir)
            except Exception as exc:
                logger.warning("tensorboard 初始化失败（%s），回退到控制台", exc)
                self._tb_writer = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info("step=%d  %s=%.6g", step, k, v)
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as exc:  # pragma: no cover
                logger.warning("wandb 写日志失败：%s", exc)
        if self._tb_writer is not None:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._tb_writer.add_scalar(k, v, global_step=step)
            except Exception as exc:  # pragma: no cover
                logger.warning("tensorboard 写日志失败：%s", exc)

    def close(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Trainer 主体
# ----------------------------------------------------------------------
class Trainer:
    """手写训练循环。

    典型用法（预训练入口）::

        model = ScratchForCausalLM(cfg.model)
        train_loader = build_pretrain_loader(cfg)
        trainer = Trainer(model, train_loader, tcfg, eval_loader=eval_loader)
        trainer.train()

    设计要点：
    - **不强制**模型返回什么：forward 输出是标量 loss 或 ``{"loss": ...}`` 字典都接受；
      这样与 ``ScratchForCausalLM``（内部用 ``causal_lm_loss``）天然对接，也便于单测里
      用最小 ``nn.Module`` 验证。
    - 单卡 / CPU 直接跑；多卡 DDP / DeepSpeed / FSDP 由 ``distributed.py`` 在外部包装
      后再传入本类（本类对"模型已经被包裹"是透明的）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        config: TrainerConfig,
        eval_loader: Optional[Iterable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[CosineWarmupScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        set_seed(config.seed)

        # ---- 设备 ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)

        # ---- 梯度检查点（优先让模型自己开） ----
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # ---- torch.compile（惰性、可选） ----
        if config.torch_compile:
            try:
                logger.info("启用 torch.compile(mode='default')")
                self.model = torch.compile(self.model)
            except Exception as exc:  # pragma: no cover
                logger.warning("torch.compile 失败（%s），跳过", exc)

        # ---- 优化器 / 调度器 ----
        self.optimizer = optimizer or build_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_eps,
            optimizer_type=config.optimizer_type,
        )
        # 总优化步数：外部传入 max_steps 优先；否则按 epochs 估算
        self.max_steps = self._resolve_max_steps(train_loader)
        self.scheduler = scheduler or build_scheduler(
            self.optimizer,
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=self.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        ensure_dir(config.output_dir)
        self._log = _LogWriter(
            config.report_to,
            project=config.project_name,
            run_name=config.run_name or Path(config.output_dir).name,
            output_dir=config.output_dir,
        )

        # 运行时状态
        self.global_step = 0
        self._loss_accum = 0.0
        self._accum_count = 0

        # ---- 断点续训 ----
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    # ------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------
    def _enable_gradient_checkpointing(self) -> None:
        """优先调用模型自带的 gradient_checkpointing_enable；没有则 warning。"""
        method = getattr(self.model, "gradient_checkpointing_enable", None)
        if callable(method):
            try:
                method()
                logger.info("已开启模型自带的梯度检查点")
                return
            except Exception as exc:
                logger.warning("调用 gradient_checkpointing_enable 失败：%s", exc)
        logger.warning(
            "模型未实现 gradient_checkpointing_enable，梯度检查点未生效；"
            "如使用 ScratchForCausalLM 请确认 model 子模块支持该方法"
        )

    def _resolve_max_steps(self, train_loader: Iterable) -> int:
        """确定总优化步数。

        - 配置里 ``max_steps > 0`` 直接用；
        - 否则按 ``len(train_loader) / GAS * num_train_epochs`` 估算。
        """
        if self.config.max_steps and self.config.max_steps > 0:
            return int(self.config.max_steps)
        try:
            steps_per_epoch = max(1, len(train_loader) // self.config.gradient_accumulation_steps)  # type: ignore[arg-type]
        except TypeError:
            # iterable 没有 len（流式 memmap），无法估算；让外部必须传 max_steps
            raise ValueError(
                "train_loader 没有长度且未配置 training.max_steps；"
                "流式数据集必须在 YAML 里显式给出 max_steps"
            )
        return int(steps_per_epoch * self.config.num_train_epochs)

    def _autocast_context(self):
        """根据 mixed_precision 返回 autocast 上下文（CPU 上回退到 nullcontext）。"""
        mp = self.config.mixed_precision.lower()
        if mp == "no" or not torch.cuda.is_available():
            # CPU / fp32 训练：空上下文
            import contextlib
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _move_batch(self, batch: Any) -> Any:
        """把 batch 里的张量搬到 device，兼容 dict / list / tuple / tensor。"""
        if isinstance(batch, dict):
            return {k: self._move_batch(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [self._move_batch(v) for v in batch]
            return type(batch)(*moved) if isinstance(batch, tuple) else moved
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch

    def _extract_loss(self, out: Any) -> torch.Tensor:
        """从模型输出里取出标量 loss。

        支持三种形态：裸张量、字典（``out["loss"]``）、带 ``loss`` 属性的输出
        容器（如 ``CausalLMOutput`` dataclass）。
        """
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            if "loss" not in out:
                raise KeyError(f"模型输出 dict 缺少 'loss' 字段，键有：{list(out.keys())}")
            return out["loss"]
        loss_attr = getattr(out, "loss", None)
        if isinstance(loss_attr, torch.Tensor):
            return loss_attr
        raise TypeError(f"无法从类型 {type(out)} 中解析 loss")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def train(self) -> None:
        """运行完整训练循环，直到 max_steps。"""
        cfg = self.config
        self.model.train()
        logger.info(
            "开始训练：global_batch_size = MBS(%d) × GAS(%d) × 卡数(1) = %d；"
            "max_steps=%d；warmup=%d；device=%s",
            cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps,
            cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            self.max_steps, cfg.warmup_steps, self.device,
        )

        accum = 0
        start = time.time()
        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                batch = self._move_batch(batch)
                with self._autocast_context():
                    out = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = self._extract_loss(out)
                    # 梯度累积时要除以 GAS，让不同 GAS 下的梯度尺度一致
                    loss = loss / cfg.gradient_accumulation_steps

                # backward：DDP / 普通 module 都透明；loss 已按 GAS 缩放
                loss.backward()

                # 记录未缩放前的 loss 用于日志
                self._loss_accum += float(loss.detach()) * cfg.gradient_accumulation_steps
                self._accum_count += 1
                accum += 1

                if accum % cfg.gradient_accumulation_steps != 0:
                    continue

                # ---- 一次优化器更新 ----
                # 1) 梯度裁剪：把全局范数限制在 max_grad_norm 内，防止 loss spike
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                # 2) optimizer.step() + scheduler.step()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # 3) 日志
                if self.global_step % cfg.logging_steps == 0:
                    avg_loss = self._loss_accum / max(1, self._accum_count)
                    self._loss_accum = 0.0
                    self._accum_count = 0
                    elapsed = time.time() - start
                    self._log.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/step": self.global_step,
                            "train/elapsed_sec": elapsed,
                        },
                        self.global_step,
                    )

                # 4) 评估
                if self.eval_loader is not None and self.global_step % cfg.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self._log.log({"eval/loss": eval_loss}, self.global_step)

                # 5) 存点
                if self.global_step % cfg.save_steps == 0:
                    self.save_checkpoint()

        # 训练结束：最后存一次
        self.save_checkpoint()
        self._log.close()
        logger.info("训练完成，共 %d 步", self.global_step)

    @torch.no_grad()
    def evaluate(self) -> float:
        """在 eval_loader 上前向一遍，返回平均 loss。"""
        if self.eval_loader is None:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.eval_loader:
            batch = self._move_batch(batch)
            with self._autocast_context():
                out = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                loss = self._extract_loss(out)
            total += float(loss.detach())
            n += 1
        self.model.train()
        return total / max(1, n)

    # ------------------------------------------------------------------
    # checkpoint 存 / 取
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        """保存一份 checkpoint：模型权重 + 优化器 + 调度器 + 步数。

        命名 ``checkpoint-{step}``；保留最近 ``save_total_limit`` 份。
        """
        cfg = self.config
        ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{self.global_step}"
        ensure_dir(ckpt_dir)

        # 模型 state_dict（DDP 包裹时取 .module）
        model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": cfg.__dict__,
        }
        torch.save(state, ckpt_dir / "training.pt")
        # 同时写一份人类可读的 meta
        with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"global_step": self.global_step, "max_steps": self.max_steps},
                f, indent=2, ensure_ascii=False,
            )
        logger.info("checkpoint 已保存：%s", ckpt_dir)
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        """只保留最近 ``save_total_limit`` 份 checkpoint。"""
        cfg = self.config
        root = Path(cfg.output_dir)
        ckpts = sorted(
            [p for p in root.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        while len(ckpts) > cfg.save_total_limit:
            old = ckpts.pop(0)
            for child in old.glob("*"):
                child.unlink()
            old.rmdir()
            logger.info("清理旧 checkpoint：%s", old)

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 目录或 ``training.pt`` 文件恢复训练状态。"""
        p = Path(path)
        if p.is_dir():
            p = p / "training.pt"
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        state = torch.load(p, map_location="cpu", weights_only=False)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        logger.info("已从 %s 恢复到 global_step=%d", p, self.global_step)

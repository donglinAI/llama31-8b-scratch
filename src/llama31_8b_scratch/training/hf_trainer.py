"""工业路径：HF Trainer / Accelerate / TRL SFTTrainer 的薄封装（全部惰性 import）。

为什么除了手写 Trainer 还要留这条路径？
- 手写 Trainer 适合教学与定制，但真机上跑 8B+ 时，成熟库已经把一堆工程细节磨平：
  DDP 启动、梯度累积、bf16 梯度缩放（fp16 时 GradScaler）、dataset pin_memory、
  dataloader 的分布式 sampler、checkpoint 容灾、与 DeepSpeed / FSDP 的开箱集成。
- TRL 的 ``SFTTrainer`` 进一步把 prompt mask（只对 assistant 算 loss）、序列打包
  （packing）、chat template 应用等封装好了，是 SFT 的工业默认路径。

本模块只做两件事：
1. 把我们的 YAML 配置翻译成 ``TrainingArguments`` / ``SFTConfig``；
2. 在函数内部惰性 import ``transformers`` / ``trl`` / ``accelerate``，
   缺库时给出清晰报错，绝不拖包子 import。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["run_sft_with_trl", "build_hf_training_arguments"]


def build_hf_training_arguments(cfg: Any) -> Any:
    """从我们的 TrainerConfig / dict 构造 ``transformers.TrainingArguments``。

    惰性 import transformers；缺库时抛出带安装提示的 RuntimeError。
    """
    try:
        from transformers import TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "transformers 未安装，HF 训练路径不可用；请 pip install 'llama31-8b-scratch[train]'"
        ) from exc

    # 兼容 Config / dict / TrainerConfig
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    per_device_bs = int(d.get("per_device_train_batch_size", 4))
    gas = int(d.get("gradient_accumulation_steps", 1))

    return TrainingArguments(
        output_dir=d.get("output_dir", "./outputs/sft"),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=gas,
        learning_rate=float(d.get("learning_rate", 2e-5)),
        weight_decay=float(d.get("weight_decay", 0.0)),
        max_grad_norm=float(d.get("max_grad_norm", 1.0)),
        num_train_epochs=float(d.get("num_train_epochs", 3.0)),
        max_steps=int(d.get("max_steps", -1)),
        warmup_steps=int(d.get("warmup_steps", 0)),
        lr_scheduler_type="cosine",
        logging_steps=int(d.get("logging_steps", 10)),
        save_steps=int(d.get("save_steps", 500)),
        evaluation_strategy="steps" if d.get("eval_loader") else "no",
        bf16=(d.get("mixed_precision", "no") == "bf16"),
        fp16=(d.get("mixed_precision", "no") == "fp16"),
        optim=d.get("optimizer_type", "adamw_torch"),
        report_to=(d.get("report_to", "none") or "none").split(","),
        gradient_checkpointing=bool(d.get("gradient_checkpointing", False)),
        torch_compile=bool(d.get("torch_compile", False)),
        dataloader_num_workers=int(d.get("dataloader_num_workers", 0)),
        save_total_limit=int(d.get("save_total_limit", 3)),
    )


def run_sft_with_trl(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    cfg: Any = None,
    peft_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """用 TRL ``SFTTrainer`` 跑 SFT（工业路径）。

    Args:
        model: 已加载的 HF 模型（或 Scratch 转 HF 权重后的模型）。
        tokenizer: HF tokenizer（负责 apply_chat_template）。
        train_dataset: HF datasets 风格的 ``Dataset``。
        eval_dataset: 可选评估集。
        cfg: TrainerConfig / dict。
        peft_config: 可选 peft.LoraConfig dict。

    Returns:
        ``SFTTrainer.train()`` 的结果。
    """
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "trl 未安装，TRL SFTTrainer 路径不可用；请 pip install 'llama31-8b-scratch[train]'"
        ) from exc

    args = build_hf_training_arguments(cfg)

    # 可选 LoRA：把 dict 翻成 peft.LoraConfig
    peft_cfg_obj = None
    if peft_config is not None:
        from peft import LoraConfig  # type: ignore

        peft_cfg_obj = LoraConfig(
            r=int(peft_config.get("r", 8)),
            lora_alpha=int(peft_config.get("lora_alpha", 16)),
            lora_dropout=float(peft_config.get("lora_dropout", 0.05)),
            target_modules=list(peft_config.get("target_modules", ["q_proj", "v_proj"])),
            bias=peft_config.get("bias", "none"),
            task_type=peft_config.get("task_type", "CAUSAL_LM"),
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg_obj,
    )
    return trainer.train()

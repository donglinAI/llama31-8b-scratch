#!/usr/bin/env python3
"""DPO 训练入口（手写训练循环，薄封装）。

手写成对损失见 :mod:`llama31_8b_scratch.align.dpo`；工业 TRL ``DPOTrainer`` 的惰性封装见
:mod:`llama31_8b_scratch.align.trl_wrappers`。本脚本默认走**手写路径**：真实构建策略/参考
模型、加载偏好数据集、计算序列对数概率并反传，CPU 即可跑通。

用法::

    python3 05_align/run_dpo.py --config configs/align/dpo_smoke.yaml
    python3 05_align/run_dpo.py --config configs/align/dpo_8b.yaml --override align.beta=0.05
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch
import torch.nn.functional as F

from llama31_8b_scratch.align.dpo import dpo_loss
from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.common.paths import resolve_path
from llama31_8b_scratch.common.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO 训练入口")
    parser.add_argument("--config", required=True, help="对齐配置 YAML 路径")
    parser.add_argument(
        "--override", action="append", default=[], help="点号路径覆盖项，可重复"
    )
    return parser.parse_args()


def build_policy_and_reference(cfg):
    """构建策略模型与冻结参考模型；二者初始权重完全相同。"""
    from llama31_8b_scratch.model import ScratchForCausalLM, ModelConfig

    model_config = ModelConfig.from_yaml_config(cfg)
    policy = ScratchForCausalLM(model_config)
    reference = ScratchForCausalLM(ModelConfig.from_yaml_config(cfg))
    # DPO 起点：参考模型 = 训练前的 SFT 模型，这里复制同一份初始权重
    reference.load_state_dict(policy.state_dict())
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()
    return policy, reference


def build_tokenizer(cfg):
    from llama31_8b_scratch.data import ScratchBPETokenizer

    return ScratchBPETokenizer.from_file(str(resolve_path(cfg.tokenizer.path)))


def build_loader(cfg, tokenizer):
    from torch.utils.data import DataLoader

    from llama31_8b_scratch.data import PreferenceCollator, PreferenceDataset

    dataset = PreferenceDataset(
        source=resolve_path(cfg.data.train_file),
        tokenizer=tokenizer,
        max_length=int(cfg.data.seq_len),
    )
    collator = PreferenceCollator(pad_token_id=int(cfg.token.pad_token_id))
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.per_device_train_batch_size),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg.training.dataloader_num_workers),
    )


def branch_sequence_logp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """计算一条支路（chosen 或 rejected）的条件序列对数概率，形状 ``[B]``。

    这里严格处理 label 移位：用 ``logits[:, :-1]`` 对齐 ``tokens[:, 1:]``，
    即只累加“在已有上下文条件下”的 token 对数概率，跳过无意义的首 token。
    """
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    valid = attention_mask[:, 1:].to(torch.bool)

    token_logp = F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    # 沿时间维把有效位置的条件对数概率求和，得到整条序列的 logp
    return (token_logp * valid).sum(dim=-1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    if cfg.get("align.method", "dpo") != "dpo":
        raise ValueError("run_dpo 期望 align.method=dpo")
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("llama31_8b_scratch.align.run_dpo")

    policy, reference = build_policy_and_reference(cfg)
    tokenizer = build_tokenizer(cfg)
    loader = build_loader(cfg, tokenizer)

    tcfg = cfg.training
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(tcfg.learning_rate),
        betas=tuple(tcfg.adam_betas),
        eps=float(tcfg.adam_eps),
        weight_decay=float(tcfg.weight_decay),
    )

    beta = float(cfg.align.beta)
    label_smoothing = float(cfg.get("align.label_smoothing", 0.0))
    max_steps = int(tcfg.max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    reference.to(device)

    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            def _t(name: str) -> torch.Tensor:
                return batch[name].to(device)

            with torch.no_grad():
                ref_chosen = branch_sequence_logp(
                    reference, _t("chosen_input_ids"), _t("chosen_attention_mask")
                )
                ref_rejected = branch_sequence_logp(
                    reference, _t("rejected_input_ids"), _t("rejected_attention_mask")
                )

            policy_chosen = branch_sequence_logp(
                policy, _t("chosen_input_ids"), _t("chosen_attention_mask")
            )
            policy_rejected = branch_sequence_logp(
                policy, _t("rejected_input_ids"), _t("rejected_attention_mask")
            )

            result = dpo_loss(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                ref_chosen_logps=ref_chosen,
                ref_rejected_logps=ref_rejected,
                beta=beta,
                label_smoothing=label_smoothing,
            )
            loss = result["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.max_grad_norm and float(tcfg.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(tcfg.max_grad_norm))
            optimizer.step()
            step += 1

            if step % int(tcfg.logging_steps) == 0 or step == 1:
                logger.info(
                    "step=%d loss=%.4f reward_margin=%.4f acc=%.2f",
                    step, float(loss.detach()),
                    float(result["rewards_margin"].mean()),
                    float(result["accuracy"]),
                )

    logger.info("DPO 训练完成，共 %d 步，权重目录：%s", step, tcfg.output_dir)


if __name__ == "__main__":
    main()

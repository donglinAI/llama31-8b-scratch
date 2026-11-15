"""预训练入口（薄封装）。

用法::

    # CPU smoke
    python 03_pretrain/run_pretrain.py --config configs/pretrain/smoke.yaml

    # 8B 真机（8 卡）
    torchrun --nproc_per_node=8 03_pretrain/run_pretrain.py \\
        --config configs/pretrain/8b.yaml

    # 命令行覆盖
    python 03_pretrain/run_pretrain.py --config configs/pretrain/smoke.yaml \\
        --override training.max_steps=100 --override training.report_to=wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---- 三行 shim：零安装也能 import llama31_8b_scratch ----
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.training.trainer import Trainer, TrainerConfig

logger = get_logger("llama31_8b_scratch.pretrain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="llama31-8b-scratch 预训练入口")
    p.add_argument("--config", required=True, help="YAML 配置路径")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="点号覆盖项，如 training.max_steps=100；可重复",
    )
    return p.parse_args()


def build_model(cfg):
    """按冻结接口构造 ScratchForCausalLM（模型代理提供）。"""
    # 模型代理提供 ModelConfig.from_yaml_config 与 ScratchForCausalLM
    # from_yaml_config 接收根配置（含 model / token 节点），不是 model 子节点
    from llama31_8b_scratch.model import ScratchForCausalLM, ModelConfig

    model_cfg = ModelConfig.from_yaml_config(cfg)
    return ScratchForCausalLM(model_cfg)


def build_tokenizer(cfg):
    """加载已训练分词器；未配置路径时在训练语料上现训一个并保存。"""
    from llama31_8b_scratch.common.paths import resolve_path
    from llama31_8b_scratch.data import ScratchBPETokenizer

    tok_path = cfg.get("tokenizer.path", None)
    if tok_path:
        return ScratchBPETokenizer.from_file(str(resolve_path(tok_path)))

    # 兜底：无现成词表，按模型词表规模在预训练语料上现训
    from llama31_8b_scratch.data.process_pretrain import read_jsonl

    records = read_jsonl(resolve_path(cfg.data.train_file))
    tokenizer = ScratchBPETokenizer.train(
        texts=[str(r["text"]) for r in records],
        vocab_size=int(cfg.model.vocab_size),
        min_frequency=1,
    )
    tokenizer.save(str(resolve_path(cfg.training.output_dir) / "tokenizer"))
    return tokenizer


def build_dataloaders(cfg, tokenizer):
    """按数据代理的真实签名构造预训练数据。"""
    from torch.utils.data import DataLoader

    from llama31_8b_scratch.common.paths import resolve_path
    from llama31_8b_scratch.data import PretrainCollator, PretrainDataset

    data_cfg = cfg.data
    train_ds = PretrainDataset(
        source=resolve_path(data_cfg.train_file),
        tokenizer=tokenizer,
        max_length=int(data_cfg.seq_len),
        memmap=bool(getattr(data_cfg, "memmap", False)),
    )
    collator = PretrainCollator(pad_token_id=int(cfg.token.pad_token_id))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.per_device_train_batch_size),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg.training.dataloader_num_workers),
    )
    return train_loader, None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    tcfg = TrainerConfig.from_config(cfg.training)
    model = build_model(cfg)
    tokenizer = build_tokenizer(cfg)
    train_loader, eval_loader = build_dataloaders(cfg, tokenizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=tcfg,
        eval_loader=eval_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()

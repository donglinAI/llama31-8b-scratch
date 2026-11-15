"""SFT 入口（薄封装）。

用法::

    # CPU smoke
    python 04_sft/run_sft.py --config configs/sft/smoke.yaml

    # 8B LoRA 真机
    torchrun --nproc_per_node=8 04_sft/run_sft.py --config configs/sft/8b.yaml

    # 走 TRL 工业路径（需装 trl/accelerate）
    python 04_sft/run_sft.py --config configs/sft/8b.yaml --backend trl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---- 三行 shim ----
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.training.trainer import Trainer, TrainerConfig

logger = get_logger("llama31_8b_scratch.sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="llama31-8b-scratch SFT 入口")
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument(
        "--backend",
        choices=["handmade", "trl"],
        default="handmade",
        help="handmade=手写训练循环（默认）；trl=TRL SFTTrainer 工业路径",
    )
    return p.parse_args()


def build_model(cfg):
    from llama31_8b_scratch.model import ScratchForCausalLM, ModelConfig

    # from_yaml_config 接收根配置（含 model / token 节点）
    model_cfg = ModelConfig.from_yaml_config(cfg)
    return ScratchForCausalLM(model_cfg)


def maybe_apply_peft(model, cfg):
    """按配置注入 LoRA/QLoRA（peft 缺失时降级为冻结 base 参数）。"""
    if not getattr(cfg, "peft", None):
        return model
    if not getattr(cfg.peft, "use_peft", False):
        return model

    from llama31_8b_scratch.training.peft_utils import LoraConfig, apply_lora

    lora_cfg = LoraConfig(
        r=int(cfg.peft.r),
        lora_alpha=int(cfg.peft.lora_alpha),
        lora_dropout=float(cfg.peft.lora_dropout),
        target_modules=list(cfg.peft.target_modules),
        use_qlora=bool(getattr(cfg.peft, "use_qlora", False)),
        bnb_4bit_quant_type=getattr(cfg.peft, "bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(cfg.peft, "bnb_4bit_compute_dtype", "bfloat16"),
    )
    model, backend = apply_lora(model, lora_cfg)
    logger.info("LoRA 后端：%s", backend)
    return model


def build_tokenizer(cfg):
    """加载已训练分词器；未配置路径时在 SFT 语料上现训一个。"""
    from llama31_8b_scratch.common.paths import resolve_path
    from llama31_8b_scratch.data import ScratchBPETokenizer

    tok_path = cfg.get("tokenizer.path", None)
    if tok_path:
        return ScratchBPETokenizer.from_file(str(resolve_path(tok_path)))

    # 兜底：从 messages 里抽取全部文本现训，词表规模与模型对齐
    from llama31_8b_scratch.data.process_pretrain import read_jsonl

    records = read_jsonl(resolve_path(cfg.data.train_file))
    texts = [
        str(m.get("content", ""))
        for r in records
        for m in r["messages"]
    ]
    tokenizer = ScratchBPETokenizer.train(
        texts=texts, vocab_size=int(cfg.model.vocab_size), min_frequency=1
    )
    tokenizer.save(str(resolve_path(cfg.training.output_dir) / "tokenizer"))
    return tokenizer


def build_dataloaders(cfg, tokenizer):
    from torch.utils.data import DataLoader

    from llama31_8b_scratch.common.paths import resolve_path
    from llama31_8b_scratch.data import SFTCollator, SFTDataset

    train_ds = SFTDataset(
        source=resolve_path(cfg.data.train_file),
        tokenizer=tokenizer,
        max_length=int(cfg.data.seq_len),
    )
    collator = SFTCollator(pad_token_id=int(cfg.token.pad_token_id))
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

    if args.backend == "trl":
        _run_trl(cfg)
        return

    tcfg = TrainerConfig.from_config(cfg.training)
    model = build_model(cfg)
    model = maybe_apply_peft(model, cfg)
    tokenizer = build_tokenizer(cfg)
    train_loader, eval_loader = build_dataloaders(cfg, tokenizer)

    trainer = Trainer(
        model=model, train_loader=train_loader, config=tcfg, eval_loader=eval_loader,
    )
    trainer.train()


def _run_trl(cfg) -> None:
    """TRL 工业路径：惰性依赖 transformers/trl。"""
    from llama31_8b_scratch.training.hf_trainer import run_sft_with_trl

    # 这里假定模型被转成 HF 格式后加载；详见 hf_trainer.py 文档
    raise NotImplementedError(
        "TRL 路径需要 HF 格式模型与 datasets 数据集；请在模型/数据代理就绪后接入。"
        "当前手写路径（--backend handmade）已可用。"
    )


if __name__ == "__main__":
    main()

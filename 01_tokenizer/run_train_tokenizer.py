"""分词器训练薄入口（编号目录脚本，实质实现在 ``src/llama31_8b_scratch/data/``）。

用法::

    python3 01_tokenizer/run_train_tokenizer.py --config configs/tokenizer/bpe_tokenizers.yaml
    python3 01_tokenizer/run_train_tokenizer.py --config <yaml> --override output_dir=out

支持后端由配置 ``backend`` 切换：tokenizers（默认，最小依赖可跑）/
sentencepiece / tiktoken；后两者缺包时仅对应路径报错，不影响默认路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from llama31_8b_scratch.common.config import load_config  # noqa: E402
from llama31_8b_scratch.common.paths import ensure_dir, project_root  # noqa: E402
from llama31_8b_scratch.data.process_pretrain import read_jsonl  # noqa: E402


def _load_corpus(paths: list[str]) -> list[str]:
    """读取语料：``.jsonl`` 取 text 字段，其他后缀按纯文本逐行读取。"""
    texts: list[str] = []
    root = project_root()
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        if p.suffix == ".jsonl":
            for record in read_jsonl(p):
                texts.append(str(record["text"]))
        else:
            texts.extend(
                line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
            )
    return texts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练分词器（BPE / SP / tiktoken）")
    parser.add_argument("--config", required=True, help="分词器 YAML 配置路径")
    parser.add_argument(
        "--override", action="append", default=[], help="a.b=value 形式覆盖，可重复"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    backend = cfg.get("backend", "tokenizers")
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/tokenizer/bpe"))
    special_tokens = list(cfg.get("special_tokens", []))

    if backend == "tokenizers":
        from llama31_8b_scratch.data.tokenizer_bpe import ScratchBPETokenizer

        train_cfg = cfg.train
        texts = _load_corpus(list(train_cfg.get("corpus_files", [])))
        tokenizer = ScratchBPETokenizer.train(
            texts=texts,
            vocab_size=int(train_cfg.get("vocab_size", 32_000)),
            min_frequency=int(train_cfg.get("min_frequency", 2)),
            max_token_length=int(train_cfg.get("max_token_length", 20)),
            special_tokens=special_tokens,
            show_progress=bool(train_cfg.get("show_progress", False)),
        )
    elif backend == "sentencepiece":
        from llama31_8b_scratch.data.tokenizer_sentencepiece import ScratchSentencePieceTokenizer

        train_cfg = cfg.train
        tokenizer = ScratchSentencePieceTokenizer.train(
            input_path=str(train_cfg.get("input_path")),
            model_prefix=str(train_cfg.get("model_prefix")),
            vocab_size=int(train_cfg.get("vocab_size", 32_000)),
            model_type=str(train_cfg.get("model_type", "bpe")),
            special_tokens=special_tokens,
            character_coverage=float(train_cfg.get("character_coverage", 0.9995)),
            num_threads=int(train_cfg.get("num_threads", 4)),
        )
    elif backend == "tiktoken":
        from llama31_8b_scratch.data.tokenizer_tiktoken import ScratchTiktokenTokenizer

        train_cfg = cfg.train
        texts = _load_corpus(list(train_cfg.get("corpus_files", [])))
        tokenizer = ScratchTiktokenTokenizer.train(
            texts=texts,
            vocab_size=int(train_cfg.get("vocab_size", 32_000)),
            special_tokens=special_tokens,
        )
    else:
        raise ValueError(f"未知后端: {backend}")

    tokenizer.save(str(output_dir))
    print(f"[tokenizer] backend={backend} vocab_size={tokenizer.vocab_size()}")
    print(f"[tokenizer] saved to: {output_dir}")
    print(f"[tokenizer] special token ids: {tokenizer.special_token_ids()}")


if __name__ == "__main__":
    main()

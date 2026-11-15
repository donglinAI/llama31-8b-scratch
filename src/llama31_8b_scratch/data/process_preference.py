"""偏好对齐数据编排脚本：jsonl 读写 + 调用纯逻辑校验清洗。

纯逻辑在 :mod:`llama31_8b_scratch.data.process_preference_core`；``datasets`` 惰性导入。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from llama31_8b_scratch.data import process_preference_core
from llama31_8b_scratch.data.process_pretrain import read_jsonl, write_jsonl


def load_with_datasets(path: str | Path, split: Optional[str] = None) -> Any:
    """可选：用 datasets 加载偏好 jsonl（惰性导入）。"""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("未安装 datasets；标准 jsonl 流程请使用 read_jsonl") from exc
    return load_dataset("json", data_files=str(path), split=split or "train")


def run(input_path: str | Path, output_path: str | Path, dedupe: bool = True) -> Dict[str, int]:
    """读取 -> 校验清洗 -> 写出，返回统计信息。"""
    records = read_jsonl(input_path)
    cleaned = process_preference_core.clean_batch(records, dedupe=dedupe)
    write_jsonl(cleaned, output_path)
    return {"raw": len(records), "kept": len(cleaned), "dropped": len(records) - len(cleaned)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="偏好 jsonl 校验清洗编排")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args(argv)
    stats = run(args.input, args.output, dedupe=not args.no_dedupe)
    print(f"[process_preference] {stats}")


if __name__ == "__main__":  # pragma: no cover
    main()

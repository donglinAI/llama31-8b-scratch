"""预训练数据编排脚本：jsonl 读写 + 调用纯逻辑清洗。

纯逻辑在 :mod:`llama31_8b_scratch.data.process_pretrain_core`（仅标准库）；本模块只做
IO 编排。``datasets`` 为可选重依赖，仅在 :func:`load_with_datasets` 内部惰性导入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data import process_pretrain_core


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """逐行读取 UTF-8 jsonl，跳过空行；非法行抛出 ``ValueError``。"""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl 第 {line_no} 行解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"jsonl 第 {line_no} 行不是对象")
            records.append(record)
    return records


def write_jsonl(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    """把记录列表写成 UTF-8 jsonl，每行一个 JSON 对象。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_with_datasets(path: str | Path, split: Optional[str] = None) -> Any:
    """可选：用 ``datasets`` 库加载（惰性导入，未安装时给出明确报错）。

    标准 jsonl 流程无需调用本函数；保留它是为了对接 HuggingFace 数据集格式。
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "未安装 datasets；标准 jsonl 流程请使用 read_jsonl，或先安装 datasets"
        ) from exc
    return load_dataset("json", data_files=str(path), split=split or "train")


def run(
    input_path: str | Path,
    output_path: str | Path,
    min_chars: int = 10,
    max_chars: int = 100_000,
    dedupe: bool = True,
) -> Dict[str, int]:
    """读取 -> 清洗 -> 写出，返回统计信息。"""
    records = read_jsonl(input_path)
    cleaned, stats = process_pretrain_core.clean_batch(
        records,
        min_chars=min_chars,
        max_chars=max_chars,
        dedupe=dedupe,
    )
    write_jsonl(cleaned, output_path)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口：``--input`` / ``--output`` 必填。"""
    parser = argparse.ArgumentParser(description="预训练 jsonl 清洗编排")
    parser.add_argument("--input", required=True, help="输入 jsonl 路径")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args(argv)

    stats = run(
        args.input,
        args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        dedupe=not args.no_dedupe,
    )
    print(f"[process_pretrain] {stats}")


if __name__ == "__main__":  # pragma: no cover
    main()

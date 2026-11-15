"""偏好对齐数据清洗纯逻辑（仅标准库，不依赖 datasets/torch）。

输入 schema（SPEC 5.3）::

    {"prompt": "用户输入文本", "chosen": "更好的回答", "rejected": "更差的回答"}

适用于 RM / DPO / ORPO 等成对偏好任务。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data import cleaning_core


def clean_record(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """清洗单条偏好记录。

    校验：三个字段均为非空字符串；归一化后 chosen 与 rejected 不能完全相同。
    """
    if not isinstance(record, dict):
        return None
    prompt = record.get("prompt")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not all(isinstance(x, str) for x in (prompt, chosen, rejected)):
        return None

    prompt = cleaning_core.normalize_text(prompt)
    chosen = cleaning_core.normalize_text(chosen)
    rejected = cleaning_core.normalize_text(rejected)
    if not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def clean_batch(
    records: Sequence[Dict[str, Any]],
    dedupe: bool = True,
) -> List[Dict[str, str]]:
    """批量清洗偏好记录，按 prompt+chosen+rejected 指纹保序去重。"""
    cleaned: List[Dict[str, str]] = []
    for record in records:
        result = clean_record(record)
        if result is not None:
            cleaned.append(result)

    if dedupe:
        seen: set[str] = set()
        unique: List[Dict[str, str]] = []
        for record in cleaned:
            fingerprint = cleaning_core.text_hash(
                f"{record['prompt']}\n{record['chosen']}\n{record['rejected']}"
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(record)
        cleaned = unique
    return cleaned

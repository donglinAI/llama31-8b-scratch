"""预训练数据清洗纯逻辑（仅标准库，不依赖 datasets/torch）。

输入 schema（SPEC 5.3）：``{"text": "原始文本"}``。
输出保持同一 schema，仅做归一化、质量过滤与去重。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data import cleaning_core


def clean_record(
    record: Dict[str, Any],
    min_chars: int = 10,
    max_chars: int = 100_000,
    strip_urls: bool = True,
    strip_html: bool = True,
) -> Optional[Dict[str, str]]:
    """清洗单条预训练记录。

    Args:
        record: 含 ``text`` 字段的字典。
        min_chars: 质量过滤最小字符数。
        max_chars: 质量过滤最大字符数。
        strip_urls: 是否移除 URL。
        strip_html: 是否移除 HTML 标签。

    Returns:
        合格记录返回 ``{"text": ...}``；不合格（缺字段、过短、乱码过多）返回 None。
    """
    if not isinstance(record, dict) or "text" not in record:
        return None
    text = record["text"]
    if not isinstance(text, str):
        return None

    if strip_html:
        text = cleaning_core.strip_html_tags(text)
    if strip_urls:
        text = cleaning_core.strip_urls(text)
    text = cleaning_core.normalize_text(text)

    if not cleaning_core.is_quality_text(
        text, min_chars=min_chars, max_chars=max_chars
    ):
        return None
    return {"text": text}


def clean_batch(
    records: Sequence[Dict[str, Any]],
    min_chars: int = 10,
    max_chars: int = 100_000,
    dedupe: bool = True,
    strip_urls: bool = True,
    strip_html: bool = True,
) -> List[Dict[str, str]]:
    """批量清洗预训练记录，返回合格记录列表与统计信息。

    Returns:
        ``(cleaned, stats)`` 二元组；``stats`` 含原始/保留/丢弃/去重计数。
    """
    cleaned: List[Dict[str, str]] = []
    dropped = 0
    for record in records:
        result = clean_record(
            record,
            min_chars=min_chars,
            max_chars=max_chars,
            strip_urls=strip_urls,
            strip_html=strip_html,
        )
        if result is None:
            dropped += 1
        else:
            cleaned.append(result)

    before_dedupe = len(cleaned)
    if dedupe:
        cleaned = cleaning_core.deduplicate_by_hash(cleaned, text_key="text")
    stats = {
        "raw": len(records),
        "kept": len(cleaned),
        "dropped": dropped,
        "duplicates_removed": before_dedupe - len(cleaned),
    }
    return cleaned, stats

"""数据清洗与去重的纯逻辑模块（仅依赖标准库，不依赖 torch 与重型数据框架）。

设计原则（SPEC 6）：纯算法逻辑尽量放在只依赖标准库/numpy 的模块里，便于在无
torch 环境下做单测。本模块提供可复用的文本归一化、质量过滤与哈希去重能力，
预训练 / SFT / 偏好三类数据的编排脚本在此基础上做各自的字段校验。
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence

# 预编译正则：URL、HTML 标签、连续空白行、行尾空白
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# 控制字符（保留 \t \n \r）
_CONTROL_CHARS = {
    chr(c) for c in range(32) if chr(c) not in ("\t", "\n", "\r")
} | {chr(127)}


def normalize_text(text: str) -> str:
    """文本归一化：Unicode NFC 合并、去控制字符、压缩空白。

    步骤：

    1. ``unicodedata.normalize("NFC", ...)`` 把兼容字符合并为标准形式，避免同一个
       字存在多种编码表示导致去重失效；
    2. 移除除 ``\\t \\n \\r`` 外的控制字符；
    3. 反转义 HTML 实体（``&amp;`` -> ``&`` 等）；
    4. 去掉行尾空白、压缩连续空格/制表符，把 3 个以上换行压成 2 个。

    Args:
        text: 原始文本。

    Returns:
        归一化后的文本；空输入返回空字符串。
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(ch for ch in text if ch not in _CONTROL_CHARS)
    text = html.unescape(text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def strip_urls(text: str) -> str:
    """删除文本中的 URL（http/https 与 www 开头）。"""
    return _URL_RE.sub("", text)


def strip_html_tags(text: str) -> str:
    """删除 HTML 标签（实体解码在 :func:`normalize_text` 中统一处理）。"""
    return _HTML_TAG_RE.sub("", text)


_WHITESPACE_RE = re.compile(r"\s+")


def text_hash(text: str) -> str:
    """对归一化后的文本计算 SHA-1 摘要，作为去重指纹。

    先做 :func:`normalize_text`，再剔除全部空白字符后哈希，使「仅空格/换行不同、
    内容相同」的文本得到同一指纹。
    """
    normalized = normalize_text(text)
    fingerprint_source = _WHITESPACE_RE.sub("", normalized)
    return hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()


def _junk_ratio(text: str) -> float:
    """统计「非字母数字、非 CJK、非常见标点」字符占比，用于低质文本过滤。"""
    if not text:
        return 1.0
    junk = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z") or cat.isspace():
            continue
        if cat.startswith("L") or cat.startswith("N"):
            continue
        junk += 1
    return junk / len(text)


def is_quality_text(
    text: str,
    min_chars: int = 10,
    max_chars: int = 100_000,
    max_junk_ratio: float = 0.3,
) -> bool:
    """质量过滤：长度区间 + 乱码占比阈值。

    Args:
        text: 待判断文本（未归一化也可，函数内部会归一化）。
        min_chars: 最少字符数。
        max_chars: 最多字符数。
        max_junk_ratio: 允许的乱码/符号字符占比上限。

    Returns:
        通过过滤返回 True。
    """
    normalized = normalize_text(text)
    if len(normalized) < min_chars or len(normalized) > max_chars:
        return False
    if _junk_ratio(normalized) > max_junk_ratio:
        return False
    return True


def deduplicate_by_hash(
    records: Sequence[Dict[str, Any]],
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """按文本哈希保序去重。

    Args:
        records: 原始记录列表。
        text_key: 参与哈希的字段名。

    Returns:
        去重后的新列表（保持首次出现顺序），不修改入参。
    """
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        content = record.get(text_key, "")
        fingerprint = text_hash(str(content))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(record)
    return unique


def normalize_batch(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量归一化（仅做文本字段清洗，不做质量/去重过滤）。"""
    return [dict(record) for record in records]

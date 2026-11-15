"""SFT 数据清洗纯逻辑（仅标准库，不依赖 datasets/torch）。

输入 schema（SPEC 5.3）::

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

校验规则：角色合法、content 非空、至少包含一轮 user 与一轮 assistant；
system 只能出现在首位。归一化只清洗正文文本，不改变消息结构。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data import cleaning_core
from llama31_8b_scratch.data.chat_template import ALLOWED_ROLES, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER


def _normalize_message(msg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """校验并归一化单条消息，非法返回 None。"""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    content = msg.get("content")
    if role not in ALLOWED_ROLES or not isinstance(content, str):
        return None
    content = cleaning_core.normalize_text(content)
    if not content:
        return None
    return {"role": role, "content": content}


def validate_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    """校验消息列表结构与角色顺序。

    规则：

    - 必须是非空列表；
    - system 若存在只能位于首位；
    - 角色序列不能连续重复（user 后不能再接 user）；
    - 至少有一条 user 与一条 assistant。

    Returns:
        归一化后的消息列表；非法返回 None。
    """
    if not isinstance(messages, list) or not messages:
        return None

    normalized: List[Dict[str, str]] = []
    prev_role: Optional[str] = None
    has_user = False
    has_assistant = False

    for idx, msg in enumerate(messages):
        cleaned = _normalize_message(msg)
        if cleaned is None:
            return None
        role = cleaned["role"]

        if idx == 0 and role == ROLE_SYSTEM:
            normalized.append(cleaned)
            prev_role = role
            continue
        if role == ROLE_SYSTEM:
            return None  # system 只能在首位
        if prev_role == role:
            return None  # 连续同角色
        if role == ROLE_USER:
            has_user = True
        elif role == ROLE_ASSISTANT:
            has_assistant = True
        normalized.append(cleaned)
        prev_role = role

    if not has_user or not has_assistant:
        return None
    return normalized


def clean_record(record: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """清洗单条 SFT 记录。"""
    if not isinstance(record, dict) or "messages" not in record:
        return None
    messages = validate_messages(record["messages"])
    if messages is None:
        return None
    return {"messages": messages}


def clean_batch(
    records: Sequence[Dict[str, Any]],
    dedupe: bool = True,
) -> List[Dict[str, List[Dict[str, str]]]]:
    """批量清洗 SFT 记录。

    去重指纹基于「角色+正文」拼接后的归一化文本。
    """
    cleaned: List[Dict[str, List[Dict[str, str]]]] = []
    for record in records:
        result = clean_record(record)
        if result is not None:
            cleaned.append(result)

    if dedupe:
        seen: set[str] = set()
        unique: List[Dict[str, List[Dict[str, str]]]] = []
        for record in cleaned:
            fingerprint = cleaning_core.text_hash(
                "".join(f"{m['role']}:{m['content']}" for m in record["messages"])
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(record)
        cleaned = unique
    return cleaned

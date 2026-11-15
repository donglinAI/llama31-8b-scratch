"""Llama 3.1 对话模板（数据代理冻结实现，下游训练/对齐代理只调用，不自行拼字符串）。

模板逐字对应 SPEC 5.4：

.. code-block:: text

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {assistant}<|eot_id|>

要点：

- 序列以 ``<|begin_of_text|>``（BOS）开头；
- 每一轮对话由「角色头 + 空行 + 正文 + ``<|eot_id|>``」组成；
- 若消息列表中没有 ``system`` 角色，则省略 system 整块；
- 本模块只负责**字符串拼接**，不做分词；分词由各 tokenizer 后端完成。

官方 Llama 3.1 词表中特殊 token 的 id 是固定常量（词表规模 128256），
本模块同时保存字符串与官方 id；自训练 BPE 词表的 id 映射由具体 tokenizer
实例通过 ``token_to_id`` 查询，模板字符串保持不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Llama 3.1 官方特殊 token：字符串与固定 id（与 configs/model/*.yaml 的 token 节点一致）
# ---------------------------------------------------------------------------

BOS_TOKEN: str = "<|begin_of_text|>"
EOS_TOKEN: str = "<|end_of_text|>"
EOT_TOKEN: str = "<|eot_id|>"
PAD_TOKEN: str = "<|finetune_right_pad_token|>"
START_HEADER_TOKEN: str = "<|start_header_id|>"
END_HEADER_TOKEN: str = "<|end_header_id|>"
UNK_TOKEN: str = "<unk>"

#: 官方词表中的固定 id（仅用于对齐 Llama3.1 原生词表；自训练词表以实例映射为准）
BOS_ID: int = 128000
EOS_ID: int = 128001
EOT_ID: int = 128009
PAD_ID: int = 128004
START_HEADER_ID: int = 128006
END_HEADER_ID: int = 128007

#: 官方词表规模
VOCAB_SIZE: int = 128256

#: 模板中允许出现的角色顺序
ROLE_SYSTEM: str = "system"
ROLE_USER: str = "user"
ROLE_ASSISTANT: str = "assistant"
ALLOWED_ROLES: tuple[str, ...] = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT)

#: 全部需要注册进分词器词表的特殊 token（顺序即推荐的注册顺序）
SPECIAL_TOKENS: tuple[str, ...] = (
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    EOT_TOKEN,
    PAD_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
)

#: 字符串 -> 官方固定 id 的映射（原生 Llama3.1 词表）
OFFICIAL_TOKEN_IDS: Dict[str, int] = {
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
    EOT_TOKEN: EOT_ID,
    PAD_TOKEN: PAD_ID,
    START_HEADER_TOKEN: START_HEADER_ID,
    END_HEADER_TOKEN: END_HEADER_ID,
}


def _validate_message(msg: Mapping[str, Any]) -> None:
    """校验单条消息字段完整、角色合法，非法时抛出 ``ValueError``。"""
    if not isinstance(msg, Mapping):
        raise ValueError(f"消息必须是字典，实际为 {type(msg)!r}")
    if "role" not in msg or "content" not in msg:
        raise ValueError(f"消息缺少 role/content 字段: {msg!r}")
    role = msg["role"]
    if role not in ALLOWED_ROLES:
        raise ValueError(f"非法角色 {role!r}，允许值: {ALLOWED_ROLES}")
    if not isinstance(msg["content"], str):
        raise ValueError(f"content 必须是字符串，实际为 {type(msg['content'])!r}")


def render_block(role: str, content: str) -> str:
    """渲染单轮对话块（不含 BOS）。

    结构为：``<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>``。

    Args:
        role: 角色（system/user/assistant）。
        content: 该轮正文。

    Returns:
        渲染后的字符串。
    """
    if role not in ALLOWED_ROLES:
        raise ValueError(f"非法角色 {role!r}")
    return (
        f"{START_HEADER_TOKEN}{role}{END_HEADER_TOKEN}\n\n"
        f"{content}{EOT_TOKEN}"
    )


def render_messages(
    messages: List[Mapping[str, Any]],
    token_ids: Optional[Mapping[str, int]] = None,
) -> str:
    """把消息列表渲染成 Llama 3.1 模板字符串。

    Args:
        messages: 形如 ``[{"role": "user", "content": "..."}]`` 的消息列表。
        token_ids: 预留参数。若传入特殊 token 字符串 -> id 的映射，本函数仍返回
            模板字符串（id 级替换由分词器在 ``encode`` 阶段完成）；保留该参数是为了
            与 SPEC 5.4 的接口签名对齐。

    Returns:
        拼接后的完整模板文本，首字符为 BOS 特殊 token。
    """
    del token_ids  # 模板层只产出文本；id 映射在分词阶段解析
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages 必须是非空列表")

    parts: List[str] = [BOS_TOKEN]
    for msg in messages:
        _validate_message(msg)
        content = msg["content"].strip()
        parts.append(render_block(msg["role"], content))
    return "".join(parts)


class LlamaChatTemplate:
    """Llama 3.1 对话模板的稳定封装，下游统一通过本类调用。

    类属性保存特殊 token 的字符串与官方 id；``apply`` 同时接受纯文本（按 user 轮
    包裹）或消息列表。训练/对齐侧不要手写模板字符串，统一走这里。
    """

    BOS: str = BOS_TOKEN
    EOS: str = EOS_TOKEN
    EOT: str = EOT_TOKEN
    PAD: str = PAD_TOKEN
    START_HEADER: str = START_HEADER_TOKEN
    END_HEADER: str = END_HEADER_TOKEN
    UNK: str = UNK_TOKEN

    BOS_ID: int = BOS_ID
    EOS_ID: int = EOS_ID
    EOT_ID: int = EOT_ID
    PAD_ID: int = PAD_ID
    START_HEADER_ID: int = START_HEADER_ID
    END_HEADER_ID: int = END_HEADER_ID

    SPECIAL_TOKENS: tuple[str, ...] = SPECIAL_TOKENS
    OFFICIAL_TOKEN_IDS: Dict[str, int] = dict(OFFICIAL_TOKEN_IDS)

    @classmethod
    def render_messages(cls, messages: List[Mapping[str, Any]]) -> str:
        """消息列表 -> 模板字符串，委托给模块级 :func:`render_messages`。"""
        return render_messages(messages)

    @classmethod
    def render_block(cls, role: str, content: str) -> str:
        """单轮块渲染，委托给 :func:`render_block`。"""
        return render_block(role, content)

    def apply(
        self,
        text: Optional[str] = None,
        messages: Optional[List[Mapping[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """对外统一入口：接受文本或消息列表，返回模板字符串。

        Args:
            text: 纯文本输入；与 ``messages`` 二选一。传入时按单轮 user 消息渲染。
            messages: 消息列表。
            add_generation_prompt: 为 True 时在末尾追加 assistant 角色头（不含正文），
                供推理阶段让模型续写；训练阶段一般为 False。

        Returns:
            渲染后的模板字符串。
        """
        if messages is not None:
            rendered = render_messages(messages)
        elif text is not None:
            rendered = render_messages(
                [{"role": ROLE_USER, "content": str(text)}]
            )
        else:
            raise ValueError("apply 需要传入 text 或 messages 之一")

        if add_generation_prompt:
            rendered += (
                f"{START_HEADER_TOKEN}{ROLE_ASSISTANT}{END_HEADER_TOKEN}\n\n"
            )
        return rendered

    @staticmethod
    def split_prompt_completion(
        messages: List[Mapping[str, Any]],
    ) -> tuple[str, str]:
        """把 SFT 消息拆成「prompt 文本」与「assistant 补全文本」。

        prompt 部分包含 BOS、system 轮与最后一个 user 轮（含其 ``<|eot_id|>``）；
        completion 为最后一条 assistant 消息正文，渲染时再补 assistant 角色头与 EOT。
        该拆分供 SFT 数据集做 prompt 部分 label 置 -100 使用。

        Returns:
            ``(prompt_text, assistant_content)`` 二元组。
        """
        if not messages:
            raise ValueError("messages 不能为空")
        if messages[-1]["role"] != ROLE_ASSISTANT:
            raise ValueError("SFT 样本最后一条必须是 assistant 消息")

        prompt_messages = messages[:-1]
        completion = messages[-1]["content"].strip()

        if prompt_messages:
            prompt_text = render_messages(prompt_messages)
        else:
            # 没有前置轮次时，prompt 至少包含 BOS 与 user 角色头由调用方补齐
            prompt_text = BOS_TOKEN
        return prompt_text, completion

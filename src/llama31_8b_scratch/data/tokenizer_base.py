"""分词器统一接口（稳定契约，下游数据集/collator 只依赖本接口）。

三种后端（tokenizers Rust BPE / SentencePiece / tiktoken）都实现同一组方法，
下游无需关心底层库。接口刻意保持精简：编码、解码、消息渲染编码、词表查询、
特殊 token id 映射与序列化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence


class ScratchTokenizer(ABC):
    """分词器抽象基类。"""

    # -- 编码 / 解码 ---------------------------------------------------------

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """把文本编码为 token id 列表。"""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """把 token id 列表还原为文本。"""

    def encode_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> List[int]:
        """把消息列表经对话模板渲染后编码。默认走 :class:`LlamaChatTemplate`。"""
        from llama31_8b_scratch.data.chat_template import LlamaChatTemplate

        rendered = LlamaChatTemplate().apply(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(rendered, add_special_tokens=False)

    # -- 词表查询 -----------------------------------------------------------

    @abstractmethod
    def vocab_size(self) -> int:
        """返回当前词表规模（含特殊 token）。"""

    @abstractmethod
    def token_to_id(self, token: str) -> Optional[int]:
        """token 字符串 -> id；不存在返回 None。"""

    @abstractmethod
    def id_to_token(self, idx: int) -> Optional[str]:
        """id -> token 字符串；不存在返回 None。"""

    @abstractmethod
    def special_token_ids(self) -> Dict[str, int]:
        """返回特殊 token 字符串 -> id 的映射。"""

    # -- 序列化 -------------------------------------------------------------

    @abstractmethod
    def save(self, directory: str) -> None:
        """把词表/模型保存到目录。"""

    @classmethod
    @abstractmethod
    def from_file(cls, path: str) -> "ScratchTokenizer":
        """从已保存的目录/文件加载分词器。"""

    # -- 便捷属性 -----------------------------------------------------------

    def bos_id(self) -> Optional[int]:
        return self.special_token_ids().get("<|begin_of_text|>")

    def eos_id(self) -> Optional[int]:
        return self.special_token_ids().get("<|end_of_text|>")

    def eot_id(self) -> Optional[int]:
        return self.special_token_ids().get("<|eot_id|>")

    def pad_id(self) -> Optional[int]:
        return self.special_token_ids().get("<|finetune_right_pad_token|>")

    def __len__(self) -> int:
        return self.vocab_size()

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"{type(self).__name__}(vocab_size={self.vocab_size()})"

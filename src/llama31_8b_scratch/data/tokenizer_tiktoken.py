"""tiktoken 后端（惰性导入，未安装时不影响包顶层 import）。

tiktoken 是 OpenAI 用于 GPT 系列的 BPE 实现，编码速度快。本模块提供两种构造方式：

- ``from_encoding``：直接复用已发布的 tiktoken 编码（如 ``o200k_base``）；
- ``train``：在自有语料上用 ``tiktoken.train_bpe`` 训练合并表后构造 Encoding。

当前最小依赖环境未安装 ``tiktoken``，smoke 测试不经过本模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data.chat_template import SPECIAL_TOKENS
from llama31_8b_scratch.data.tokenizer_base import ScratchTokenizer


class ScratchTiktokenTokenizer(ScratchTokenizer):
    """tiktoken 封装。"""

    def __init__(self, encoding: Any, special_tokens: Sequence[str]) -> None:
        self._enc = encoding
        self._special = list(special_tokens)

    @classmethod
    def from_encoding(
        cls,
        encoding_name: str = "o200k_base",
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "ScratchTiktokenTokenizer":
        """按名称加载内置 tiktoken 编码。"""
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding(encoding_name)
        return cls(encoding, special_tokens=special_tokens)

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int = 32_000,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        pat_str: Optional[str] = None,
    ) -> "ScratchTiktokenTokenizer":
        """在自有语料上训练 BPE 合并表并构造 Encoding。

        Args:
            texts: 训练文本列表。
            vocab_size: 目标词表规模。
            special_tokens: 特殊 token，以高偏移量注册。
            pat_str: 正则切分模式；缺省使用 tiktoken 默认字节级模式。
        """
        import tiktoken  # type: ignore

        merges = tiktoken.train_bpe(list(texts), vocab_size=vocab_size)
        # 特殊 token 分配到普通词表之后的高偏移位
        special_set = {
            tok: len(merges) + i for i, tok in enumerate(special_tokens)
        }
        # 通用 Unicode 切分模式：字母/数字/标点/符号/空白各自成组
        default_pat = r"\p{L}+|\p{N}+|\p{P}+|\p{S}+|\p{Z}+|\p{C}+"
        encoding = tiktoken.Encoding(
            name="scratch-tiktoken",
            pat_str=pat_str or default_pat,
            mergeable_ranks=merges,
            special_tokens=special_set,
        )
        return cls(encoding, special_tokens=special_tokens)

    @classmethod
    def from_file(cls, path: str) -> "ScratchTiktokenTokenizer":
        """tiktoken 编码通常按名称加载；文件路径保留为扩展入口。"""
        return cls.from_encoding(encoding_name=Path(path).name)

    def save(self, directory: str) -> None:
        """导出合并表与特殊 token 映射为 JSON，便于复现。"""
        import json

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self._enc.name,
            "special_tokens": self.special_token_ids(),
        }
        (out / "tiktoken.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        allowed = set(self._special)
        ids: List[int] = self._enc.encode(text, allowed_special=allowed)
        if add_special_tokens:
            bos = self.token_to_id("<|begin_of_text|>")
            if bos is not None:
                ids = [bos] + ids
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))

    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def token_to_id(self, token: str) -> Optional[int]:
        return self._enc.encode_single_token(token) if token in self._enc._special_tokens else None

    def id_to_token(self, idx: int) -> Optional[str]:
        try:
            return self._enc.decode([idx])
        except Exception:  # pragma: no cover - 越界保护
            return None

    def special_token_ids(self) -> Dict[str, int]:
        return {
            tok: idx
            for tok, idx in self._enc._special_tokens.items()
            if tok in self._special
        }

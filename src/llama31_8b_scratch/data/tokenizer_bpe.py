"""基于 ``tokenizers``（Rust BPE）的默认分词器后端。

这是唯一在最小依赖环境下即可真实训练、真实跑通的 smoke 路径（SPEC 硬约束）。
底层使用 HuggingFace ``tokenizers`` 库的字节级 BPE：

- 归一化：Unicode NFC；
- 预分词：字节级（ByteLevel），可覆盖任意字节，无需为未知字符准备 unk 回退之外的处理；
- 训练：BPE 迭代合并高频子词对；
- 解码：字节级还原原始 UTF-8。

SentencePiece / tiktoken 后端见同目录其他模块，均为惰性导入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data.chat_template import SPECIAL_TOKENS
from llama31_8b_scratch.data.tokenizer_base import ScratchTokenizer


class ScratchBPETokenizer(ScratchTokenizer):
    """tokenizers Rust BPE 封装。"""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    # -- 训练 / 加载 / 保存 ------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Optional[Sequence[str]] = None,
        files: Optional[Sequence[str]] = None,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        max_token_length: int = 20,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        show_progress: bool = False,
    ) -> "ScratchBPETokenizer":
        """从文本流或文件训练 BPE 词表。

        Args:
            texts: 内存中的文本列表，与 ``files`` 至少给一个。
            files: 纯文本文件路径列表，按行流式读取。
            vocab_size: 目标词表规模（不含/含特殊 token 以库行为为准）。
            min_frequency: 子词对进入合并表的最低词频。
            max_token_length: 单个 token 最大字符长度。
            special_tokens: 注册的特殊 token 列表，按顺序分配低位 id。
            show_progress: 是否显示训练进度条。
        """
        from tokenizers import Tokenizer  # type: ignore
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore
        from tokenizers.models import BPE  # type: ignore
        from tokenizers.normalizers import NFC  # type: ignore
        from tokenizers.pre_tokenizers import ByteLevel  # type: ignore
        from tokenizers.trainers import BpeTrainer  # type: ignore

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            special_tokens=list(special_tokens),
            show_progress=show_progress,
        )

        if files:
            tokenizer.train(list(files), trainer=trainer)
        elif texts:
            tokenizer.train_from_iterator(list(texts), trainer=trainer)
        else:
            raise ValueError("train 需要提供 texts 或 files 之一")

        # 二次登记特殊 token，保证编码时整串识别、不被预分词切开
        tokenizer.add_special_tokens(list(special_tokens))
        return cls(tokenizer)

    @classmethod
    def from_file(cls, path: str) -> "ScratchBPETokenizer":
        """从 ``tokenizer.json`` 或保存目录加载。"""
        from tokenizers import Tokenizer  # type: ignore

        p = Path(path)
        if p.is_dir():
            p = p / "tokenizer.json"
        return cls(Tokenizer.from_file(str(p)))

    def save(self, directory: str) -> None:
        """保存 ``tokenizer.json`` 与特殊 token 映射 ``special_tokens.json``。"""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out / "tokenizer.json"))
        (out / "special_tokens.json").write_text(
            json.dumps(self.special_token_ids(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- 编码 / 解码 ---------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本；``add_special_tokens`` 时在序列头部补 BOS。"""
        ids: List[int] = self._tok.encode(text, add_special_tokens=False).ids
        if add_special_tokens:
            bos = self.token_to_id("<|begin_of_text|>")
            if bos is not None:
                ids = [bos] + ids
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    # -- 词表查询 ------------------------------------------------------------

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._tok.id_to_token(idx)

    def special_token_ids(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            idx = self.token_to_id(token)
            if idx is not None:
                mapping[token] = idx
        return mapping

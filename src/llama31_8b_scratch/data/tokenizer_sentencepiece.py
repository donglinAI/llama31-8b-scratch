"""SentencePiece BPE 后端（惰性导入，未安装时不影响包顶层 import）。

SentencePiece 是 Llama 系列原生使用的分词方案：直接在原始字节/字符流上建模，
不依赖空格预分词，对中文等无空格语言友好。本模块把训练与推理封装到统一的
:class:`~llama31_8b_scratch.data.tokenizer_base.ScratchTokenizer` 接口。

注意：当前最小依赖环境未安装 ``sentencepiece``，本模块只在方法内部 import，
smoke 测试走 :mod:`llama31_8b_scratch.data.tokenizer_bpe`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama31_8b_scratch.data.chat_template import SPECIAL_TOKENS
from llama31_8b_scratch.data.tokenizer_base import ScratchTokenizer


class ScratchSentencePieceTokenizer(ScratchTokenizer):
    """SentencePiece BPE 封装。"""

    def __init__(self, processor: Any, special_tokens: Sequence[str]) -> None:
        self._sp = processor
        self._special = list(special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str,
        model_prefix: str,
        vocab_size: int = 32_000,
        model_type: str = "bpe",
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        character_coverage: float = 0.9995,
        num_threads: int = 4,
    ) -> "ScratchSentencePieceTokenizer":
        """训练 SentencePiece 模型并返回实例。

        Args:
            input_path: 纯文本语料文件路径（每行一条文本）。
            model_prefix: 模型输出前缀，会生成 ``.model`` / ``.vocab``。
            vocab_size: 词表规模。
            model_type: ``bpe`` / ``unigram`` / ``char`` / ``word``。
            special_tokens: 控制符号列表。
            character_coverage: 字符覆盖率（中文语料建议 0.9995 以上）。
            num_threads: 训练线程数。
        """
        import sentencepiece as spm  # type: ignore

        Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=input_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            control_symbols=list(special_tokens),
            character_coverage=character_coverage,
            num_threads=num_threads,
            pad_id=-1,  # 不强制占用 pad id；pad 由特殊 token 映射管理
            bos_id=1,
            eos_id=2,
            unk_id=0,
        )
        processor = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        return cls(processor, special_tokens=special_tokens)

    @classmethod
    def from_file(cls, path: str) -> "ScratchSentencePieceTokenizer":
        """加载 ``.model`` 文件。"""
        import sentencepiece as spm  # type: ignore

        processor = spm.SentencePieceProcessor(model_file=path)
        return cls(processor, special_tokens=SPECIAL_TOKENS)

    def save(self, directory: str) -> None:
        """把底层模型拷贝到目录（SentencePiece 模型在训练时已落盘）。"""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        # processor 持有模型文件路径时可直接拷贝；这里以序列化方式兜底
        self._sp.Save(str(out / "spm.model"))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_special_tokens:
            bos = self.token_to_id("<|begin_of_text|>")
            if bos is not None:
                ids = [bos] + ids
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._sp.decode(list(ids))

    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def token_to_id(self, token: str) -> Optional[int]:
        idx = self._sp.PieceToId(token)
        return idx if idx >= 0 else None

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._sp.IdToPiece(idx)

    def special_token_ids(self) -> Dict[str, int]:
        return {tok: idx for tok in self._special if (idx := self.token_to_id(tok)) is not None}

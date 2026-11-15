"""数据子包：分词器、对话模板、清洗编排、数据集与 collator、序列打包。

顶层 import 仅依赖核心依赖（torch/numpy/yaml）；sentencepiece / tiktoken / datasets
等可选重依赖均在对应函数内部惰性导入（SPEC 3）。
"""

from __future__ import annotations

from llama31_8b_scratch.data.chat_template import (
    BOS_TOKEN,
    EOS_TOKEN,
    EOT_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
    LlamaChatTemplate,
    render_block,
    render_messages,
)
from llama31_8b_scratch.data.collators import (
    PreferenceCollator,
    PretrainCollator,
    SFTCollator,
)
from llama31_8b_scratch.data.datasets import (
    GRPODataset,
    PreferenceDataset,
    PretrainDataset,
    SFTDataset,
)
from llama31_8b_scratch.data.packing import (
    concatenate_sequences,
    pack_blocks,
    pack_blocks_with_labels,
)
from llama31_8b_scratch.data.tokenizer_base import ScratchTokenizer
from llama31_8b_scratch.data.tokenizer_bpe import ScratchBPETokenizer
from llama31_8b_scratch.data.tokenizer_extend import expand_embedding, extend_vocab
from llama31_8b_scratch.data.tokenizer_sentencepiece import ScratchSentencePieceTokenizer
from llama31_8b_scratch.data.tokenizer_tiktoken import ScratchTiktokenTokenizer

__all__ = [
    # 模板
    "BOS_TOKEN",
    "EOS_TOKEN",
    "EOT_TOKEN",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "START_HEADER_TOKEN",
    "END_HEADER_TOKEN",
    "LlamaChatTemplate",
    "render_block",
    "render_messages",
    # 数据集
    "PretrainDataset",
    "SFTDataset",
    "PreferenceDataset",
    "GRPODataset",
    # collator
    "PretrainCollator",
    "SFTCollator",
    "PreferenceCollator",
    # 打包
    "concatenate_sequences",
    "pack_blocks",
    "pack_blocks_with_labels",
    # 分词器
    "ScratchTokenizer",
    "ScratchBPETokenizer",
    "ScratchSentencePieceTokenizer",
    "ScratchTiktokenTokenizer",
    # 词表扩充
    "extend_vocab",
    "expand_embedding",
]

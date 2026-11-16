"""词元嵌入层 token embedding 手写实现。

形状约定
--------
- 输入 input_ids: [batch, seq_len]，元素为词表下标；
- 输出 [batch, seq_len, hidden_size];
- 参数 weight: [vocab_size, hidden_size]，键名路径为
  model.embed_tokens.weight （与 HF Llama 逐字一致）。
"""

from __future__ import annotations

import torch
from torch import nn


class ScratchEmbedding(nn.Module):
    """可学习词元嵌入矩阵，初始化遵循配置中的 initializer_range。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 直接持有 weight 参数，保证 state_dict 键名与 HF 对齐；
        # 这里不使用 nn.Embedding，避免额外的 bias/scale 包装层。
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """按下标查表，返回嵌入向量序列。"""
        return torch.nn.functional.embedding(input_ids, self.weight)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"

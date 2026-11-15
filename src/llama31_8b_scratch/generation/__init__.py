"""生成与解码策略包：KV Cache、采样过滤、自回归生成。

对外暴露：

- :class:`KVCache`：按层存储的动态键值缓存；
- :func:`generate`：阻塞式自回归生成；
- :func:`stream_generate`：流式生成器；
- 采样工具：:func:`sample_next_token`、:func:`top_k_filter`、
  :func:`top_p_filter`、:func:`apply_repetition_penalty`。
"""

from __future__ import annotations

from llama31_8b_scratch.generation.cache import KVCache
from llama31_8b_scratch.generation.generate import generate, stream_generate
from llama31_8b_scratch.generation.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "KVCache",
    "generate",
    "stream_generate",
    "sample_next_token",
    "top_k_filter",
    "top_p_filter",
    "apply_temperature",
    "apply_repetition_penalty",
]

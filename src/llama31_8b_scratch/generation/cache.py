"""动态 KV Cache 手写实现。

背景
----
自回归解码时，每生成一个新 token，都需要前面所有 token 的 K/V 向量。
如果每次都把整段历史重新前向，复杂度随长度平方增长。KV Cache 把每层
历史 K/V 缓存下来，解码步只需计算当前新 token 的 K/V 并追加到缓存尾部，
注意力计算从 ``O(T^2 · d)`` 降到 ``O(T · d)`` 每步。

数据布局
--------
每层缓存两个张量：

- ``key_cache[layer]``: ``[B, num_kv_heads, T, head_dim]``；
- ``value_cache[layer]``: ``[B, num_kv_heads, T, head_dim]``。

``update`` 在对应层尾部按序列维度（dim=2）拼接新的 K/V。
"""

from __future__ import annotations

from typing import List, Optional

import torch


class KVCache:
    """按层存储的动态键值缓存。"""

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """追加当前步的 K/V 到指定层缓存，返回该层最新完整 K/V。

        Args:
            key_states: ``[B, num_kv_heads, 1, head_dim]``（解码步）或
                ``[B, num_kv_heads, T, head_dim]``（prefill）。
            value_states: 形状与 ``key_states`` 相同。
            layer_idx: 层下标。

        Returns:
            ``(cached_keys, cached_values)``，已拼接历史与当前步。
        """
        if layer_idx == len(self.key_cache):
            # 该层首次写入：直接保存当前张量
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 后续步：沿序列维度 dim=2 拼接
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """返回指定层缓存的序列长度；缓存为空时返回 0。"""
        if len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def __len__(self) -> int:
        """已缓存的层数。"""
        return len(self.key_cache)

    def clear(self) -> None:
        """清空全部层缓存（复用同一实例做新一轮生成时调用）。"""
        self.key_cache.clear()
        self.value_cache.clear()

    def batch_select(self, indices: torch.Tensor) -> "KVCache":
        """按 batch 下标挑选子批次缓存（束搜索 / 分支生成预留接口）。"""
        new_cache = KVCache()
        for k_cache, v_cache in zip(self.key_cache, self.value_cache):
            new_cache.key_cache.append(k_cache.index_select(0, indices))
            new_cache.value_cache.append(v_cache.index_select(0, indices))
        return new_cache

    def to_legacy_tuple(self) -> Optional[tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """导出为 ``(key_cache, value_cache)`` 二元组（对齐 HF 旧接口习惯）。"""
        if not self.key_cache:
            return None
        return self.key_cache, self.value_cache

"""序列打包（sequence packing）纯逻辑模块（仅标准库）。

为什么需要打包：逐条样本训练时，短样本之间用 padding 补齐，算力浪费在 pad 上；
打包把多条样本首尾相接，切成长度固定的块（block），块内尽量不插 pad，显著提升
GPU 有效算力占用。块与块之间仍由注意力掩码/位置编码区分；跨块的上下文边界由
EOS 标记提示模型。

本模块只做「拼接 + 定长切分 + 对齐 label」，不涉及注意力实现。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def concatenate_sequences(
    sequences: Sequence[Sequence[int]],
    eos_id: Optional[int] = None,
) -> List[int]:
    """把多条序列首尾相接，可选在每条末尾补 EOS。

    Args:
        sequences: token id 列表的列表。
        eos_id: 文档分隔符 id；为 None 时不补。

    Returns:
        拼接后的一维 id 列表。
    """
    merged: List[int] = []
    for seq in sequences:
        merged.extend(seq)
        if eos_id is not None:
            merged.append(eos_id)
    return merged


def pack_blocks(
    ids: Sequence[int],
    block_size: int,
    pad_id: int,
) -> List[List[int]]:
    """把一维 id 流切成长度 ``block_size`` 的定长块，末尾不足补 pad。

    Args:
        ids: 拼接好的 token id 流。
        block_size: 每个块的序列长度。
        pad_id: 填充 id。

    Returns:
        定长块列表；若输入为空返回空列表。
    """
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    blocks: List[List[int]] = []
    for start in range(0, len(ids), block_size):
        block = list(ids[start : start + block_size])
        if len(block) < block_size:
            block.extend([pad_id] * (block_size - len(block)))
        blocks.append(block)
    return blocks


def pack_blocks_with_labels(
    ids: Sequence[int],
    labels: Sequence[int],
    block_size: int,
    pad_id: int,
    label_pad: int = -100,
) -> Tuple[List[List[int]], List[List[int]]]:
    """id 与 label 同步打包，保证两者逐位置对齐。

    Args:
        ids: token id 流。
        labels: 与 ``ids`` 等长的 label 流（prompt 位置为 -100）。
        block_size: 块长度。
        pad_id: id 填充值。
        label_pad: label 填充值（默认 -100，被交叉熵损失忽略）。

    Returns:
        ``(id_blocks, label_blocks)``，两个列表等长、逐块等长。
    """
    if len(ids) != len(labels):
        raise ValueError(f"ids({len(ids)}) 与 labels({len(labels)}) 长度不一致")
    id_blocks = pack_blocks(ids, block_size, pad_id)
    label_blocks: List[List[int]] = []
    for start in range(0, len(labels), block_size):
        block = list(labels[start : start + block_size])
        if len(block) < block_size:
            block.extend([label_pad] * (block_size - len(block)))
        label_blocks.append(block)
    return id_blocks, label_blocks


def count_tokens(sequences: Sequence[Sequence[int]]) -> int:
    """统计多条序列的 token 总数（不含分隔符）。"""
    return sum(len(seq) for seq in sequences)

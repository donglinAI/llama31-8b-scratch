"""词表扩充与 embedding 扩展（纯 numpy 核心，torch 版本惰性导入）。

场景：基座词表不含领域术语 / 新增特殊 token 时，需要在不重训 BPE 的前提下
扩充词表，并同步扩大输入 embedding（与输出 lm_head，若未绑定权重）。

数学约定：

- 旧 embedding 权重形状 ``(V_old, D)`` 原样保留；
- 新增 ``V_new - V_old`` 行，按截断正态 ``N(0, initializer_range^2)`` 初始化，
  与模型 ``initializer_range`` 配置一致；
- 扩展后形状 ``(V_new, D)``，旧行索引不变，保证已有权重可继续加载。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def extend_vocab(
    base_tokens: Sequence[str],
    new_tokens: Sequence[str],
) -> Tuple[List[str], Dict[str, int], List[str]]:
    """合并基础词表与待新增 token，去重并保持顺序。

    Args:
        base_tokens: 原有词表（token 字符串列表）。
        new_tokens: 希望加入的新 token。

    Returns:
        ``(new_vocab, token_to_id, added_tokens)``：
        - ``new_vocab``: 合并后的完整词表；
        - ``token_to_id``: token -> 新 id 映射；
        - ``added_tokens``: 实际新增（原本不存在）的 token 列表。
    """
    seen: set[str] = set()
    vocab: List[str] = []
    for token in list(base_tokens) + list(new_tokens):
        if token not in seen:
            seen.add(token)
            vocab.append(token)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    base_set = set(base_tokens)
    added = [tok for tok in new_tokens if tok not in base_set]
    return vocab, token_to_id, added


def expand_embedding(
    weight: np.ndarray,
    num_new: int,
    initializer_range: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """在旧 embedding 下方拼接新初始化行（numpy 实现）。

    Args:
        weight: 旧权重，形状 ``(V_old, D)``。
        num_new: 新增行数。
        initializer_range: 截断正态标准差。
        seed: 随机种子，保证可复现。

    Returns:
        扩展后的权重，形状 ``(V_old + num_new, D)``；``num_new <= 0`` 时原样返回副本。
    """
    if weight.ndim != 2:
        raise ValueError(f"embedding 权重必须是二维，实际形状 {weight.shape}")
    if num_new <= 0:
        return weight.copy()

    rng = np.random.default_rng(seed)
    v_old, dim = weight.shape
    new_rows = rng.normal(loc=0.0, scale=initializer_range, size=(num_new, dim))
    # 截断到 ±2σ，避免极端初始化值
    new_rows = np.clip(new_rows, -2 * initializer_range, 2 * initializer_range)
    return np.concatenate([weight.astype(np.float32), new_rows.astype(np.float32)], axis=0)


def expand_embedding_torch(
    weight: Any,
    num_new: int,
    initializer_range: float = 0.02,
    seed: int = 42,
) -> Any:
    """torch 版本：在 ``nn.Embedding.weight`` 上原地扩展（惰性导入 torch）。

    旧参数保留，新增行用 ``torch.nn.init.normal_`` 初始化；返回扩展后的参数张量。
    """
    import torch  # type: ignore

    if weight.ndim != 2:
        raise ValueError("embedding 权重必须是二维")
    if num_new <= 0:
        return weight.detach().clone()

    v_old, dim = weight.shape
    gen = torch.Generator().manual_seed(seed)
    new_rows = torch.normal(
        mean=0.0, std=initializer_range, size=(num_new, dim), generator=gen
    ).to(weight.dtype)
    new_rows = torch.clamp(new_rows, -2 * initializer_range, 2 * initializer_range)
    return torch.cat([weight.detach(), new_rows], dim=0)

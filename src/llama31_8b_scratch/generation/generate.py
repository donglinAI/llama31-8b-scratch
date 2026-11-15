"""自回归文本生成手写实现。

生成流程
--------
1. **Prefill**：一次性前向整段 prompt，得到最后位置的词表 logits，并（可选）
   填充 KV Cache；
2. **Decode 循环**：每步采样一个新 token 追加到序列；若遇到 ``eos_token_id``
   则提前停止；否则把新 token（配合缓存）送入模型，得到下一步 logits；
3. **终止条件**：达到 ``max_new_tokens``，或整批全部生成结束 token。

提供两种调用形态：

- :func:`generate`：阻塞式，返回最终完整序列；
- :func:`stream_generate`：流式生成器，逐步产出 ``token`` / ``sequence`` /
  ``step`` / ``finished``，便于边生成边渲染（SSE、终端逐字输出等）。

注意：本模块只依赖 ``torch`` 与本包内部模块；``flash_attn`` 等重依赖仍在
模型内部惰性导入，生成路径不直接触碰。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import torch

from llama31_8b_scratch.generation.cache import KVCache
from llama31_8b_scratch.generation.sampling import sample_next_token


@torch.no_grad()
def _run_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sampling_kwargs: Dict[str, Any],
    eos_token_id: Optional[int],
    use_cache: bool,
) -> Iterator[Dict[str, Any]]:
    """解码循环内部生成器，逐步产出当前步状态。

    Args:
        model: 具备 ``forward(input_ids, position_ids, past_key_values,
            use_cache)`` 接口的因果 LM。
        input_ids: ``[B, T]`` prompt 序列。
        max_new_tokens: 最多新增 token 数。
        sampling_kwargs: 透传给 :func:`sample_next_token` 的采样参数。
        eos_token_id: 结束 token id；``None`` 表示不按结束 token 停止。
        use_cache: 是否启用 KV Cache。
    """
    bsz = input_ids.shape[0]
    device = input_ids.device
    past = KVCache() if use_cache else None

    # Prefill：整段 prompt 一次前向，初始化缓存
    outputs = model(input_ids, past_key_values=past, use_cache=use_cache)
    next_logits = outputs.logits[:, -1, :]

    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    sequence = input_ids

    for step in range(max_new_tokens):
        # 1) 采样下一个 token
        next_token = sample_next_token(
            next_logits, input_ids=sequence, **sampling_kwargs
        )
        sequence = torch.cat([sequence, next_token], dim=1)

        # 2) 结束判定：本批全部命中 eos 才停止
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

        yield {
            "token": next_token,
            "sequence": sequence,
            "step": step + 1,
            "finished": bool(finished.all()),
        }

        if bool(finished.all()):
            break

        # 3) 解码步：有缓存只喂新 token；无缓存则整段重算
        if use_cache:
            position_ids = torch.full(
                (bsz, 1), past.get_seq_length(), dtype=torch.long, device=device
            )
            outputs = model(
                next_token,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
        else:
            outputs = model(sequence)
        next_logits = outputs.logits[:, -1, :]


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """阻塞式生成，返回最终完整序列。

    Args:
        model: 因果 LM 模型。
        input_ids: ``[B, T]`` prompt。
        max_new_tokens: 最多新增 token 数。
        temperature: 采样温度，``0`` 退化为贪心。
        top_k: top-k 候选数，``0`` 关闭。
        top_p: 核采样阈值，``1.0`` 关闭。
        repetition_penalty: 重复惩罚系数，``1.0`` 关闭。
        greedy: 强制贪心解码。
        eos_token_id: 结束 token id。
        pad_token_id: 填充 token id（预留，当前生成不主动填充）。
        use_cache: 是否启用 KV Cache。

    Returns:
        ``[B, T + N]`` 完整序列张量。
    """
    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "greedy": greedy,
    }
    final_sequence = input_ids
    for state in _run_decode(
        model, input_ids, max_new_tokens, sampling_kwargs, eos_token_id, use_cache
    ):
        final_sequence = state["sequence"]
    return final_sequence


def stream_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> Iterator[Dict[str, Any]]:
    """流式生成器，逐步产出状态字典。

    Yields:
        ``dict``，包含：
        - ``token``: ``[B, 1]`` 新生成 token；
        - ``sequence``: ``[B, T+step]`` 当前完整序列；
        - ``step``: 当前步数（从 1 开始）；
        - ``finished``: 本批是否全部结束。
    """
    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "greedy": greedy,
    }
    yield from _run_decode(
        model, input_ids, max_new_tokens, sampling_kwargs, eos_token_id, use_cache
    )

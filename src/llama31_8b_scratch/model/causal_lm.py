r"""因果语言模型封装（ScratchForCausalLM）手写实现。

在 :class:`ScratchModel` 顶部接一个词表投影头，输出下一个词元的 logits；
损失在模块内部完成标签移位（与 SPEC 5.5 口径一致）：

.. math::

    \\mathcal{L} = -\\sum_{t=1}^{T-1}
        \\log P(y_{t+1} \\mid y_{\\le t}; \\theta)

即 ``cross_entropy(logits[..., :-1, :], labels[..., 1:])``，
``ignore_index=-100`` 的位置不计入损失。

权重落盘兼容
------------
``save_pretrained`` 同时写出：
- ``model.safetensors``：state_dict 键名与 HF Llama 逐字一致；
- ``config.json``：``model_type="llama"``，可被 transformers / vLLM / SGLang
  按原生 Llama 结构加载，无需注册自定义模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from llama31_8b_scratch.common.logging_utils import get_logger
from llama31_8b_scratch.model.base import ScratchModel
from llama31_8b_scratch.model.config import ModelConfig

logger = get_logger("llama31_8b_scratch.model.causal_lm")


@dataclass
class CausalLMOutput:
    """因果 LM 前向输出容器。"""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    past_key_values: Optional[Any] = None


class ScratchForCausalLM(nn.Module):
    """解码器 + 词表投影头，支持权重落盘与回填。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ScratchModel(config)
        self.vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            # 绑定嵌入与输出头：不单独持有 lm_head 参数，复用 embed_tokens.weight
            self.lm_head: Optional[nn.Linear] = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

    # ------------------------------------------------------------------
    # 嵌入访问（对齐 HF 接口习惯）
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens  # type: ignore[return-value]

    def get_output_embeddings(self) -> Optional[nn.Linear]:
        return self.lm_head

    # ------------------------------------------------------------------
    # 前向与损失
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutput:
        """前向传播，返回 logits 与可选损失。

        Args:
            input_ids: ``[B, T]`` 词元下标。
            labels: ``[B, T]`` 标签，与输入等长；移位在内部完成。
            attention_mask: padding 掩码，``[B, T]``。
            position_ids: 位置编号，``[B, T]``。
            past_key_values: KV 缓存。
            use_cache: 是否更新缓存。
            gradient_checkpointing: 训练梯度检查点开关。
        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            gradient_checkpointing=gradient_checkpointing,
        )
        hidden_states = outputs.last_hidden_state

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # 绑定权重时用嵌入矩阵转置做投影：logits = hidden · W^T
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # 移位：预测位置 t 的下一个词，用 logits[:, :-1] 对齐 labels[:, 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    # ------------------------------------------------------------------
    # 权重落盘 / 回填
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str | Path) -> Path:
        """写 safetensors 权重与标准 HF ``config.json``。

        Args:
            save_directory: 输出目录，不存在会递归创建。

        Returns:
            最终目录的 ``Path``。
        """
        from safetensors.torch import save_file

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # state_dict 已按 HF 键名组织；绑定权重时不会出现 lm_head.weight
        save_file(self.state_dict(), save_dir / "model.safetensors")

        config_dict = self.config.to_hf_dict()
        (save_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return save_dir

    def load_weights(self, path: str | Path, strict: bool = False) -> None:
        """从 safetensors 回填权重。

        Args:
            path: 权重文件路径或包含 ``model.safetensors`` 的目录。
            strict: ``True`` 时要求键完全一致；``False`` 时允许绑定权重导致
                的 ``lm_head.weight`` 缺失。
        """
        from safetensors.torch import load_file

        path = Path(path)
        weight_path = path / "model.safetensors" if path.is_dir() else path
        loaded = load_file(str(weight_path))

        missing, unexpected = self.load_state_dict(loaded, strict=False)
        # 绑定词嵌入时 lm_head.weight 本就不存在，属于预期缺失
        expected_missing = {"lm_head.weight"} if self.lm_head is None else set()
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            logger.warning("权重回填存在缺失键: %s", real_missing)
        if unexpected:
            logger.warning("权重回填存在多余键: %s", unexpected)
        if strict and (real_missing or unexpected):
            raise RuntimeError(
                f"严格加载失败：missing={real_missing}, unexpected={unexpected}"
            )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        """统计参数量（评测/报告用）。"""
        params = self.parameters()
        if only_trainable:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

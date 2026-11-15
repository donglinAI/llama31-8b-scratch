r"""序列分类 / 奖励模型头（ScratchForSequenceClassification）手写实现。

在 :class:`ScratchModel` 顶部加一个标量打分头 ``score``：

.. math::

    s(x) = w^T \\, \\mathrm{Pool}(h)

其中 Pool 为按有效 token 的掩码平均池化（mean pooling），输出形状
``[B, num_labels]``；奖励模型通常 ``num_labels=1``，得到每条序列一个标量奖励。

Pairwise 打分
--------------
对齐阶段需要对同一条 prompt 的 ``chosen`` / ``rejected`` 分别打分再比较，
本模块提供 :meth:`score_pair`，输入两条序列编码，返回各自标量分，
供 RM 训练、DPO/GRPO 等对齐算法调用。

打分头权重键名：``score.weight``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import nn

from llama31_8b_scratch.model.base import ScratchModel
from llama31_8b_scratch.model.config import ModelConfig


@dataclass
class SequenceClassifierOutput:
    """序列分类前向输出容器。"""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class ScratchForSequenceClassification(nn.Module):
    """带标量奖励头的序列分类模型。"""

    def __init__(self, config: ModelConfig, num_labels: int = 1) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = ScratchModel(config)
        # 打分头：hidden_size -> num_labels，无 bias（奖励模型惯例）
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)

    # ------------------------------------------------------------------
    # 池化
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """对有效 token 做平均池化。

        Args:
            hidden_states: ``[B, T, H]``。
            attention_mask: ``[B, T]``，1 有效、0 padding；可空（视为全有效）。

        Returns:
            ``[B, H]``。
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B,T,1]
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        """前向打分。

        Args:
            input_ids: ``[B, T]``。
            attention_mask: padding 掩码，``[B, T]``。
            position_ids: 位置编号，``[B, T]``。
            labels: 可选分类标签，用于交叉熵损失（奖励训练一般不传）。
        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        pooled = self._masked_mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.score(pooled)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits, labels.view(-1), ignore_index=-100
            )
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def score_pair(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pairwise 打分：分别返回 chosen / rejected 的标量奖励。

        Returns:
            ``(reward_chosen, reward_rejected)``，形状均为 ``[B, num_labels]``。
        """
        chosen_out = self.forward(chosen_ids, attention_mask=chosen_mask)
        rejected_out = self.forward(rejected_ids, attention_mask=rejected_mask)
        return chosen_out.logits, rejected_out.logits

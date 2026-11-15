"""Batch 对齐 collator（SPEC 5.5 冻结命名）。

职责：把变长样本补齐到 batch 内等长，生成 attention mask；label 填充值统一为
-100（被交叉熵损失忽略）。序列移位仍在模型/loss 内部完成，collator 不做移位。

- :class:`PretrainCollator`：预训练等长补齐；
- :class:`SFTCollator`：SFT 样本（prompt 部分 label 已在数据集内置 -100）；
- :class:`PreferenceCollator`：chosen / rejected 两条支路分别补齐。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def _pad_1d(
    ids: Sequence[int],
    length: int,
    pad_value: int,
    padding_side: str = "right",
) -> List[int]:
    """把一维序列补齐到指定长度。"""
    seq = list(ids)
    pad_len = length - len(seq)
    if pad_len <= 0:
        return seq[:length]
    padding = [pad_value] * pad_len
    if padding_side == "right":
        return seq + padding
    return padding + seq


class PretrainCollator:
    """预训练 batch 对齐器。"""

    def __init__(
        self,
        pad_token_id: int,
        label_pad: int = -100,
        padding_side: str = "right",
        max_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad = label_pad
        self.padding_side = padding_side
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, feat in enumerate(features):
            ids = feat["input_ids"][:max_length] if self.max_length else feat["input_ids"]
            lab = feat["labels"][:max_length] if self.max_length else feat["labels"]
            ids_padded = _pad_1d(ids.tolist(), max_len, self.pad_token_id, self.padding_side)
            lab_padded = _pad_1d(lab.tolist(), max_len, self.label_pad, self.padding_side)
            input_ids[i] = torch.tensor(ids_padded, dtype=torch.long)
            labels[i] = torch.tensor(lab_padded, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class SFTCollator(PretrainCollator):
    """SFT batch 对齐器；行为与预训练一致，语义上区分 prompt 掩码来源。"""


class PreferenceCollator:
    """成对偏好 batch 对齐器，chosen / rejected 支路分别补齐。"""

    def __init__(
        self,
        pad_token_id: int,
        label_pad: int = -100,
        padding_side: str = "right",
        max_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad = label_pad
        self.padding_side = padding_side
        self.max_length = max_length

    def _pad_branch(
        self,
        features: List[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        key_ids = f"{prefix}_input_ids"
        key_labels = f"{prefix}_labels"
        key_mask = f"{prefix}_attention_mask"

        max_len = max(len(f[key_ids]) for f in features)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, feat in enumerate(features):
            ids = feat[key_ids][: self.max_length] if self.max_length else feat[key_ids]
            lab = feat[key_labels][: self.max_length] if self.max_length else feat[key_labels]
            ids_padded = _pad_1d(ids.tolist(), max_len, self.pad_token_id, self.padding_side)
            lab_padded = _pad_1d(lab.tolist(), max_len, self.label_pad, self.padding_side)
            input_ids[i] = torch.tensor(ids_padded, dtype=torch.long)
            labels[i] = torch.tensor(lab_padded, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return {
            key_ids: input_ids,
            key_labels: labels,
            key_mask: attention_mask,
        }

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        chosen = self._pad_branch(features, "chosen")
        rejected = self._pad_branch(features, "rejected")
        return {**chosen, **rejected}

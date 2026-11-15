"""数据集实现（SPEC 5.5 冻结命名）。

四类数据集：

- :class:`PretrainDataset`：支持 memmap ``.bin`` 与 jsonl 两种来源；
- :class:`SFTDataset`：消息模板渲染，prompt 部分 label 置 -100；
- :class:`PreferenceDataset`：成对偏好（chosen/rejected），供 RM/DPO/ORPO；
- :class:`GRPODataset`：prompt + 可验证标准答案，供 GRPO 规则奖励。

统一约定（SPEC 5.5）：

- 每条样本返回与 ``input_ids`` **等长**的 ``labels``；序列移位由模型/loss 内部
  用 ``F.cross_entropy(logits[..., :-1, :], labels[..., 1:])`` 完成，数据集不做移位；
- 标签填充值 -100 表示被损失忽略；
- 本模块顶层依赖 torch（核心依赖，非可选重依赖）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from llama31_8b_scratch.data.chat_template import (
    LlamaChatTemplate,
    render_block,
    render_messages,
)
from llama31_8b_scratch.data.process_pretrain import read_jsonl


def _as_long_tensor(ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.long)


class PretrainDataset(Dataset):
    """预训练数据集，支持 memmap 二进制与 jsonl。

    Args:
        source: ``.bin``（memmap）或 ``.jsonl`` 路径。
        tokenizer: 实现统一接口的分词器实例。
        max_length: 单条样本最大 token 长度。
        memmap: 显式指定是否走 memmap；None 时按扩展名判断。
        dtype: memmap 元素类型，默认 int32。
    """

    def __init__(
        self,
        source: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        memmap: Optional[bool] = None,
        dtype: str = "int32",
    ) -> None:
        self.source = Path(source)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = LlamaChatTemplate()

        use_memmap = memmap if memmap is not None else self.source.suffix == ".bin"
        self._use_memmap = use_memmap
        self._records: List[Dict[str, Any]] = []
        self._mm: Optional[np.memmap] = None

        if use_memmap:
            if not self.source.is_file():
                raise FileNotFoundError(f"memmap 文件不存在: {self.source}")
            self._mm = np.memmap(self.source, dtype=np.dtype(dtype), mode="r")
        else:
            self._records = read_jsonl(self.source)

    def __len__(self) -> int:
        if self._use_memmap:
            assert self._mm is not None
            return max(0, len(self._mm) // self.max_length)
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._use_memmap:
            assert self._mm is not None
            start = index * self.max_length
            ids = [int(x) for x in self._mm[start : start + self.max_length]]
        else:
            record = self._records[index]
            ids = self.tokenizer.encode(record["text"], add_special_tokens=True)
            ids = ids[: self.max_length]

        labels = list(ids)  # 等长 label；移位在 loss 内部完成
        return {
            "input_ids": _as_long_tensor(ids),
            "labels": _as_long_tensor(labels),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }


class SFTDataset(Dataset):
    """监督微调数据集：prompt 部分 label 置 -100。

    样本 schema（SPEC 5.3）::

        {"messages": [{"role": ..., "content": ...}, ...]}
    """

    def __init__(
        self,
        source: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.source = Path(source)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = LlamaChatTemplate()
        self._records = read_jsonl(self.source)

    def __len__(self) -> int:
        return len(self._records)

    def _encode_sft(
        self, messages: List[Dict[str, str]]
    ) -> tuple[List[int], List[int]]:
        """拆分 prompt / completion 并分别编码，返回对齐的 ids 与 labels。"""
        prompt_text, completion = self.template.split_prompt_completion(messages)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_block = render_block("assistant", completion)
        completion_ids = self.tokenizer.encode(
            completion_block, add_special_tokens=False
        )

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)

        # 超长截断：保留 completion 尾部语义；prompt 超长则右侧截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        return input_ids, labels

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        messages = self._records[index]["messages"]
        input_ids, labels = self._encode_sft(messages)
        return {
            "input_ids": _as_long_tensor(input_ids),
            "labels": _as_long_tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class PreferenceDataset(Dataset):
    """成对偏好数据集（RM / DPO / ORPO）。

    样本 schema（SPEC 5.3）::

        {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(
        self,
        source: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.source = Path(source)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = LlamaChatTemplate()
        self._records = read_jsonl(self.source)

    def __len__(self) -> int:
        return len(self._records)

    def _pair_ids(self, prompt: str, answer: str) -> List[int]:
        prompt_text = render_messages([{"role": "user", "content": prompt}])
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(
            render_block("assistant", answer), add_special_tokens=False
        )
        ids = prompt_ids + answer_ids
        return ids[: self.max_length]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self._records[index]
        chosen_ids = self._pair_ids(record["prompt"], record["chosen"])
        rejected_ids = self._pair_ids(record["prompt"], record["rejected"])
        return {
            "chosen_input_ids": _as_long_tensor(chosen_ids),
            "rejected_input_ids": _as_long_tensor(rejected_ids),
            "chosen_labels": _as_long_tensor(chosen_ids),
            "rejected_labels": _as_long_tensor(rejected_ids),
            "chosen_attention_mask": torch.ones(len(chosen_ids), dtype=torch.long),
            "rejected_attention_mask": torch.ones(len(rejected_ids), dtype=torch.long),
        }


class GRPODataset(Dataset):
    """GRPO 数据集：prompt 消息 + 可验证标准答案。

    样本 schema（SPEC 5.3）::

        {"prompt": [{"role": "user", "content": "..."}], "answer": "标准答案"}
    """

    def __init__(
        self,
        source: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.source = Path(source)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = LlamaChatTemplate()
        self._records = read_jsonl(self.source)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._records[index]
        prompt_text = render_messages(record["prompt"])
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = prompt_ids[: self.max_length]
        return {
            "prompt_input_ids": _as_long_tensor(prompt_ids),
            "prompt_attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "answer": record["answer"],
        }

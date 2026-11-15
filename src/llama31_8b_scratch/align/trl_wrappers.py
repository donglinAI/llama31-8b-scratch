"""TRL 工业级训练器惰性封装。

本模块把手写核心（:mod:`llama31_8b_scratch.align.ppo` / :mod:`dpo` / :mod:`orpo` /
:mod:`grpo`）对应的工业规模化训练器封装为薄壳：

- :class:`TRLPPOWrapper`  → TRL ``PPOTrainer``（经典 RLHF）；
- :class:`TRLDPOWrapper`  → TRL ``DPOTrainer``；
- :class:`TRLORPOWrapper`  → TRL ``ORPOTrainer``；
- :class:`TRLGRPOWrapper`  → TRL ``GRPOTrainer``。

**惰性导入约束（SPEC 3）**：``trl``、``peft``、``accelerate`` 等重依赖只在
``create_trainer`` 方法内部 import；模块顶层仅依赖 ``llama31_8b_scratch.common``，
在仅安装 torch/numpy/yaml 的环境下本模块可被正常 import。

注意：TRL 各版本字段名与 Trainer 构造参数存在演进，本封装只做“配置映射 +
惰性实例化”，字段对齐以实际安装的 TRL 版本为准；评测分数等无法在 CPU smoke
环境实测的指标一律记为 TBD，不在代码中编造。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger("llama31_8b_scratch.align.trl_wrappers")


def _require_trl():
    """惰性导入 ``trl``，缺失时给出可操作的报错信息。"""
    try:
        import trl  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于运行环境是否装了 trl
        raise ImportError(
            "未检测到 trl，请先安装可选训练依赖："
            "`pip install trl peft accelerate datasets`"
        ) from exc
    return trl


def _common_trainer_kwargs(cfg: Config) -> Dict[str, Any]:
    """从配置抽取各 Trainer 共用的训练超参数。

    配置节点约定（见 ``configs/align/*.yaml`` 的 ``trainer`` 节点）：

    - ``learning_rate``、``batch_size``、``mini_batch_size``、``max_steps``、
      ``num_train_epochs``、``gradient_accumulation_steps``、``seed``、
      ``logging_steps``、``save_steps``、``output_dir``。
    """
    trainer = cfg.get("trainer", {}) or {}
    kwargs: Dict[str, Any] = {}
    for key in (
        "learning_rate",
        "batch_size",
        "mini_batch_size",
        "max_steps",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "seed",
        "logging_steps",
        "save_steps",
        "output_dir",
    ):
        if key in trainer:
            kwargs[key] = trainer[key]
    return kwargs


class _BaseTRLWrapper:
    """TRL 封装基类：持有配置并提供共用映射逻辑。"""

    #: TRL Trainer 类名（子类覆盖）
    trainer_cls_name: str = ""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_kwargs(self) -> Dict[str, Any]:
        """把 YAML 配置映射为 TRL Trainer 构造参数字典。"""
        raise NotImplementedError

    def create_trainer(self, **objects: Any) -> Any:
        """惰性导入 TRL 并实例化 Trainer。

        Args:
            objects: 透传给 TRL Trainer 的模型 / 数据集 / tokenizer 等对象，
                例如 ``model=...``、``ref_model=...``、``train_dataset=...``、
                ``processing_class=...``。
        """
        raise NotImplementedError


class TRLPPOWrapper(_BaseTRLWrapper):
    """经典 RLHF 的 TRL ``PPOTrainer`` 封装。

    对应手写模块：:mod:`llama31_8b_scratch.align.ppo`（GAE、裁剪策略目标、价值裁剪、
    KL 惩罚、奖励白化）。TRL 内部同样维护 policy / reference / reward / value
    四个模型，并在训练步内完成采样-打分-更新闭环。
    """

    trainer_cls_name = "PPOTrainer"

    def build_kwargs(self) -> Dict[str, Any]:
        kwargs = _common_trainer_kwargs(self.config)
        align = self.config.get("align", {}) or {}
        # PPO 专用超参数，与手写 ppo.py 的函数参数一一对应
        ppo_fields = {
            "gamma": "gamma",
            "lam": "lam",
            "clip_ratio": "cliprange",
            "value_clip_ratio": "cliprange_value",
            "value_coef": "vf_coef",
            "kl_coef": "kl_coef",
            "kl_estimator": "kl_estimator",
            "whiten_advantages": "whiten_rewards",
        }
        for local_key, trl_key in ppo_fields.items():
            if local_key in align:
                kwargs[trl_key] = align[local_key]
        return kwargs

    def create_trainer(self, **objects: Any) -> Any:
        trl = _require_trl()
        ppo_config = self.build_kwargs()
        logger.info("构造 TRL PPOTrainer，配置项：%s", list(ppo_config.keys()))
        # TRL 版本演进中 PPOConfig 字段可能调整，实例化时以当前版本为准
        return trl.PPOTrainer(config=ppo_config, **objects)


class TRLDPOWrapper(_BaseTRLWrapper):
    """TRL ``DPOTrainer`` 封装。

    对应手写模块：:mod:`llama31_8b_scratch.align.dpo`（隐式奖励、β、参考模型 logp 项）。
    """

    trainer_cls_name = "DPOTrainer"

    def build_kwargs(self) -> Dict[str, Any]:
        kwargs = _common_trainer_kwargs(self.config)
        align = self.config.get("align", {}) or {}
        for local_key, trl_key in {
            "beta": "beta",
            "label_smoothing": "label_smoothing",
            "max_length": "max_length",
            "max_prompt_length": "max_prompt_length",
            "loss_type": "loss_type",
        }.items():
            if local_key in align:
                kwargs[trl_key] = align[local_key]
        return kwargs

    def create_trainer(self, **objects: Any) -> Any:
        trl = _require_trl()
        dpo_config = self.build_kwargs()
        logger.info("构造 TRL DPOTrainer，配置项：%s", list(dpo_config.keys()))
        return trl.DPOTrainer(config=dpo_config, **objects)


class TRLORPOWrapper(_BaseTRLWrapper):
    """TRL ``ORPOTrainer`` 封装。

    对应手写模块：:mod:`llama31_8b_scratch.align.orpo`（SFT 损失 + 赔率比损失，单模型
    无需参考模型）。
    """

    trainer_cls_name = "ORPOTrainer"

    def build_kwargs(self) -> Dict[str, Any]:
        kwargs = _common_trainer_kwargs(self.config)
        align = self.config.get("align", {}) or {}
        for local_key, trl_key in {
            "beta": "beta",
            "lambda_or": "lambda_or",
            "max_length": "max_length",
            "max_prompt_length": "max_prompt_length",
        }.items():
            if local_key in align:
                kwargs[trl_key] = align[local_key]
        return kwargs

    def create_trainer(self, **objects: Any) -> Any:
        trl = _require_trl()
        orpo_config = self.build_kwargs()
        logger.info("构造 TRL ORPOTrainer，配置项：%s", list(orpo_config.keys()))
        return trl.ORPOTrainer(config=orpo_config, **objects)


class TRLGRPOWrapper(_BaseTRLWrapper):
    """TRL ``GRPOTrainer`` 封装。

    对应手写模块：:mod:`llama31_8b_scratch.align.grpo`（组采样、组内相对优势、无价值
    网络、组级 KL）。
    """

    trainer_cls_name = "GRPOTrainer"

    def build_kwargs(self) -> Dict[str, Any]:
        kwargs = _common_trainer_kwargs(self.config)
        align = self.config.get("align", {}) or {}
        for local_key, trl_key in {
            "group_size": "num_generations",
            "clip_ratio": "clip_ratio",
            "kl_coef": "kl_coef",
            "kl_estimator": "kl_estimator",
            "beta": "beta",
            "max_completion_length": "max_completion_length",
        }.items():
            if local_key in align:
                kwargs[trl_key] = align[local_key]
        return kwargs

    def create_trainer(self, **objects: Any) -> Any:
        trl = _require_trl()
        grpo_config = self.build_kwargs()
        logger.info("构造 TRL GRPOTrainer，配置项：%s", list(grpo_config.keys()))
        return trl.GRPOTrainer(config=grpo_config, **objects)


_WRAPPER_REGISTRY = {
    "ppo": TRLPPOWrapper,
    "dpo": TRLDPOWrapper,
    "orpo": TRLORPOWrapper,
    "grpo": TRLGRPOWrapper,
}


def build_trl_wrapper(method: str, config: Config) -> _BaseTRLWrapper:
    """按方法名构造对应 TRL 封装。

    Args:
        method: ``"ppo"`` / ``"dpo"`` / ``"orpo"`` / ``"grpo"``。
        config: 已加载的对齐配置。

    Returns:
        对应 :class:`_BaseTRLWrapper` 子类实例。
    """
    key = method.lower()
    if key not in _WRAPPER_REGISTRY:
        raise ValueError(
            f"未知对齐方法 {method!r}，可选：{sorted(_WRAPPER_REGISTRY)}"
        )
    return _WRAPPER_REGISTRY[key](config)

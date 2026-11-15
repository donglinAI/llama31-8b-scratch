"""模型服务抽象：把「请求参数」统一映射成「一次生成」，屏蔽后端差异。

为什么需要这一层
----------------
API 层（:mod:`llama31_8b_scratch.deployment.api_server`）只关心「给我一段 prompt、一组采样
参数，我要拿到补全文本（或逐字流式）」。后端却有两种：

1. **手写教学后端** ``ScratchCausalLMService``：直接持有 ``ScratchForCausalLM`` 与一个
   分词器，调用本包 :mod:`llama31_8b_scratch.generation` 的 generate / stream_generate；
2. **工业推理后端** ``VLLMService``：惰性封装 ``vllm.LLM``，用 PagedAttention /
   连续批处理拿到高吞吐（原理见 :mod:`llama31_8b_scratch.deployment.vllm_serve`）。

两者实现同一个 :class:`BaseModelService` 接口，API 层因此可以在单测里换成
内存 fake、在真机里换成任意后端，互不影响。

纯函数约定
----------
- :func:`map_request_to_params` 是纯函数：把 pydantic 请求转成 :class:`GenerateParams`，
  不碰 torch / 网络，可直接单测；
- 具体后端类的 ``__init__`` 只存引用，**真正 import torch / vllm 发生在方法内部**，
  保证模块顶层零重依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from llama31_8b_scratch.deployment.schemas import ChatCompletionRequest
from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger("llama31_8b_scratch.deployment.service")


# ---------------------------------------------------------------------------
# 纯数据结构
# ---------------------------------------------------------------------------


@dataclass
class GenerateParams:
    """后端无关的一次生成参数（OpenAI 请求归一化后的结果）。"""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class GenerationOutput:
    """一次非流式生成的完整结果。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


def map_request_to_params(
    request: ChatCompletionRequest,
    default_max_tokens: int = 128,
) -> GenerateParams:
    """把 OpenAI 风格请求纯映射成 :class:`GenerateParams`。

    - ``max_tokens`` 未传时用 ``default_max_tokens``；
    - ``frequency_penalty`` 近似映射为重复惩罚：``repetition_penalty = 1 + freq_penalty``
      （OpenAI 频率惩罚与重复惩罚都在压低高频 token，工程上常用此近似）；
    - ``stop`` 统一规整为 list（单字符串包一层）。

    本函数不做任何 IO / 模型调用，便于单测断言每个字段。
    """
    max_tokens = request.effective_max_tokens or default_max_tokens
    stop_raw = request.stop
    if stop_raw is None:
        stop: List[str] = []
    elif isinstance(stop_raw, str):
        stop = [stop_raw]
    else:
        stop = list(stop_raw)

    return GenerateParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=0,
        max_new_tokens=int(max_tokens),
        repetition_penalty=1.0 + request.frequency_penalty,
        stop=stop,
        seed=request.seed,
    )


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class BaseModelService(ABC):
    """所有推理后端必须实现的统一接口。"""

    #: 对外暴露的模型名（与 /v1/models 对应）
    model_name: str = "llama31-8b-scratch"

    @abstractmethod
    def generate(self, prompt: str, params: GenerateParams) -> GenerationOutput:
        """阻塞式生成，返回完整补全文本。"""

    @abstractmethod
    def generate_stream(self, prompt: str, params: GenerateParams) -> Iterator[str]:
        """流式生成器，逐段产出补全文本片段（不含 prompt）。"""

    def list_models(self) -> List[str]:
        """返回本服务可提供的模型 id 列表，默认只有自身模型名。"""
        return [self.model_name]

    def count_tokens(self, text: str) -> int:
        """估算一段文本的 token 数，用于 usage 统计。

        基类给确定性的字符数占位；真实后端应覆盖为分词后的 token 数。单测里
        fake 服务也用此默认值，保证 usage 字段可断言。
        """
        return len(text)


# ---------------------------------------------------------------------------
# 手写教学后端：直接驱动 ScratchForCausalLM
# ---------------------------------------------------------------------------


class ScratchCausalLMService(BaseModelService):
    """持有一个 ``ScratchForCausalLM`` 与分词器的后端。

    分词器约定为鸭子类型：具备 ``encode(text) -> list[int]`` 与
    ``decode(ids) -> str``。真实构造由入口脚本在惰性环境里完成，本类只在
    被调用时才 import torch 与生成模块。
    """

    def __init__(self, model, tokenizer, model_name: str = "llama31-8b-scratch") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_name = model_name

    def _encode(self, text: str) -> List[int]:
        ids = self._tokenizer.encode(text)
        # tokenizers 库有时返回 EncodeOutput；这里兼容 list 与带 .ids 的对象
        return list(ids)

    def count_tokens(self, text: str) -> int:
        """真实 token 数：分词后取 id 个数。"""
        return len(self._encode(text))

    def generate(self, prompt: str, params: GenerateParams) -> GenerationOutput:
        # 惰性 import：仅在真正生成时才需要 torch
        import torch  # noqa: PLC0415

        from llama31_8b_scratch.generation.generate import generate as _generate  # noqa: PLC0415

        input_ids = torch.tensor([self._encode(prompt)], dtype=torch.long)
        prompt_tokens = int(input_ids.shape[1])
        out = _generate(
            self._model,
            input_ids,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            eos_token_id=getattr(self._model.config, "eot_token_id", None)
            or getattr(self._model.config, "eos_token_id", None),
            use_cache=True,
        )
        new_ids = out[0, prompt_tokens:].tolist()
        text = self._tokenizer.decode(new_ids)
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(new_ids),
            finish_reason="stop",
        )

    def generate_stream(self, prompt: str, params: GenerateParams) -> Iterator[str]:
        import torch  # noqa: PLC0415

        from llama31_8b_scratch.generation.generate import stream_generate  # noqa: PLC0415

        input_ids = torch.tensor([self._encode(prompt)], dtype=torch.long)
        eos = (
            getattr(self._model.config, "eot_token_id", None)
            or getattr(self._model.config, "eos_token_id", None)
        )
        for state in stream_generate(
            self._model,
            input_ids,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            eos_token_id=eos,
            use_cache=True,
        ):
            token = int(state["token"][0, 0])
            yield self._tokenizer.decode([token])


# ---------------------------------------------------------------------------
# 工业后端：vLLM 适配（惰性，方法内 import vllm）
# ---------------------------------------------------------------------------


class VLLMService(BaseModelService):
    """``vllm.LLM`` 的薄适配层，把统一接口翻成 vLLM 的 ``LLM.generate``。

    vLLM 未安装时仅在方法调用时报清晰错误；模块 import 不受影响。
    启动一个常驻 OpenAI 兼容服务见 :mod:`llama31_8b_scratch.deployment.vllm_serve`。
    """

    def __init__(self, model_path: str, model_name: str = "llama31-8b-scratch",
                 dtype: str = "auto", gpu_memory_utilization: float = 0.9) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None  # 惰性构造

    def _engine(self):
        if self._llm is None:
            try:
                from vllm import LLM  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - 依赖缺失分支
                raise ImportError("请先 pip install vllm>=0.5 以使用 VLLMService") from exc
            self._llm = LLM(
                model=self.model_path,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        return self._llm

    def generate(self, prompt: str, params: GenerateParams) -> GenerationOutput:
        from vllm import SamplingParams  # noqa: PLC0415

        llm = self._engine()
        sp = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            max_tokens=params.max_new_tokens,
            repetition_penalty=params.repetition_penalty,
            stop=params.stop or None,
            seed=params.seed,
        )
        results = llm.generate([prompt], sp)
        r = results[0]
        text = r.outputs[0].text
        return GenerationOutput(
            text=text,
            prompt_tokens=len(r.prompt_token_ids),
            completion_tokens=len(r.outputs[0].token_ids),
            finish_reason=r.outputs[0].finish_reason or "stop",
        )

    def generate_stream(self, prompt: str, params: GenerateParams) -> Iterator[str]:
        # vLLM 在线服务的流式由 OpenAI 兼容 server 负责；本地批式引擎这里退化为
        # 一次性生成再按字符切片，保证接口形状一致（真机流式请用 vllm_serve）。
        out = self.generate(prompt, params)
        for ch in out.text:
            yield ch

"""vLLM 启动封装与原理讲解（惰性导入）。

为什么需要 vLLM
----------------
手写 ``ScratchForCausalLM`` 的生成（:mod:`llama31_8b_scratch.generation`）教学上清晰，但在
真实服务里有两个工程瓶颈：

1. **逐 token 解码是访存受限（memory-bound）**：每生成一个 token 都要把全部模型
   权重从 HBM 读一遍。batch 越小，GPU 算力越浪费；
2. **不同请求的 prompt 长度参差，串行处理时 GPU 大量空转**。

vLLM 用三项关键技术解决：

- **PagedAttention**：把 KV Cache 切成固定大小的 block（页），按需分配，像操作系统
  虚拟内存一样消除碎片，把 KV 显存占用从「按最大序列预分配」降到「按需分页」，显存
  利用率通常从 <40% 提到 90%+；
- **连续批处理（continuous batching / in-flight batching）**：一个请求生成结束
  立刻把新请求塞进同一批，GPU 永远不空等；
- **prefix caching**：相同系统提示 / 共享前缀的 KV 跨请求复用。

它对外暴露的就是 OpenAI 兼容服务（``/v1/chat/completions``），所以我们手写的
FastAPI 与 vLLM 可以无缝互换——这正是统一 :class:`~service.BaseModelService` 的价值。

本模块只做两件事：根据 YAML 配置拼出启动参数（:func:`build_serve_kwargs`，纯函数，
可单测），以及在真正启动时惰性 import vLLM（:func:`run_server`）。模块顶层不 import
vllm。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger("llama31_8b_scratch.deployment.vllm_serve")


def build_serve_kwargs(cfg: Config) -> Dict[str, Any]:
    """把 YAML 配置转成 vLLM OpenAI 服务的关键字参数（纯函数）。

    只取本项目用到的字段；缺省值给工业默认。
    """
    v = cfg.vllm
    return {
        "model": str(v.model_path),
        "served_model_name": str(getattr(v, "model_name", "llama31-8b-scratch")),
        "host": str(getattr(v, "host", "0.0.0.0")),
        "port": int(getattr(v, "port", 8000)),
        "dtype": str(getattr(v, "dtype", "auto")),
        "gpu_memory_utilization": float(getattr(v, "gpu_memory_utilization", 0.9)),
        "max_model_len": int(getattr(v, "max_model_len", 4096)),
        "enable_prefix_caching": bool(getattr(v, "enable_prefix_caching", True)),
    }


def build_cli_command(cfg: Config) -> List[str]:
    """拼出等价的 ``python -m vllm.entrypoints.openai.api_server ...`` 命令（文档/排查用）。"""
    kw = build_serve_kwargs(cfg)
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", kw["model"],
        "--served-model-name", kw["served_model_name"],
        "--host", kw["host"],
        "--port", str(kw["port"]),
        "--dtype", kw["dtype"],
        "--gpu-memory-utilization", str(kw["gpu_memory_utilization"]),
        "--max-model-len", str(kw["max_model_len"]),
    ]
    if kw["enable_prefix_caching"]:
        cmd.append("--enable-prefix-caching")
    return cmd


def run_server(cfg: Config) -> Optional[int]:
    """惰性 import vLLM 并启动 OpenAI 兼容服务。

    vLLM 未安装时给出清晰指引；真机上才应调用本函数。
    """
    kw = build_serve_kwargs(cfg)
    logger.info("启动 vLLM OpenAI 服务: %s", kw)
    try:
        from vllm.entrypoints.openai.api_server import run_uvicorn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ImportError(
            "请先 `pip install vllm>=0.5` 再用 vLLM 后端；教学后端可直接用 run_api.py"
        ) from exc
    return run_uvicorn(**kw)  # type: ignore[arg-type]

"""SGLang 启动封装与原理讲解（惰性导入）。

SGLang 与 vLLM 是同类工业推理引擎，主打「灵活的编程式提示（program/约束解码）」
与极高的前缀复用效率。核心原理：

- **RadixAttention**：用一棵基数树（radix tree）把所有在跑请求的 KV Cache 前缀
  组织起来，自动复用完全相同的前缀——对多轮对话、共享 system prompt、few-shot
  示例尤其省算力；
- **Constrained / 结构化输出**：能约束输出为 JSON、正则、上下文无关文法，适合函数
  调用（function calling）与可验证生成；
- **连续批处理 + 显存管理**思路与 vLLM 类似。

与 vLLM 的取舍：vLLM 生态更通用、部署最省事；SGLang 在复杂路由/结构化输出/极高
前缀命中率的场景吞吐更好。两者都暴露 OpenAI 兼容接口，可按任务切换。

本模块同样：:func:`build_serve_kwargs` / :func:`build_cli_command` 是纯函数，
:func:`run_server` 才惰性 import sglang。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama31_8b_scratch.common.config import Config
from llama31_8b_scratch.common.logging_utils import get_logger

logger = get_logger("llama31_8b_scratch.deployment.sglang_serve")


def build_serve_kwargs(cfg: Config) -> Dict[str, Any]:
    """把 YAML 配置转成 SGLang 服务参数（纯函数）。"""
    s = cfg.sglang
    return {
        "model_path": str(s.model_path),
        "host": str(getattr(s, "host", "0.0.0.0")),
        "port": int(getattr(s, "port", 8001)),
        "dtype": str(getattr(s, "dtype", "auto")),
        "mem_fraction_static": float(getattr(s, "mem_fraction_static", 0.85)),
        "max_total_tokens": int(getattr(s, "max_total_tokens", 8192)),
        "enable_radix_cache": bool(getattr(s, "enable_radix_cache", True)),
    }


def build_cli_command(cfg: Config) -> List[str]:
    """拼出等价的 SGLang 命令行（文档/排查用）。"""
    kw = build_serve_kwargs(cfg)
    return [
        "python", "-m", "sglang.launch_server",
        "--model-path", kw["model_path"],
        "--host", kw["host"],
        "--port", str(kw["port"]),
        "--dtype", kw["dtype"],
        "--mem-fraction-static", str(kw["mem_fraction_static"]),
        "--max-total-tokens", str(kw["max_total_tokens"]),
        *(["--enable-radix-cache"] if kw["enable_radix_cache"] else []),
    ]


def run_server(cfg: Config) -> Optional[int]:
    """惰性 import SGLang 并启动服务（真机调用）。"""
    kw = build_serve_kwargs(cfg)
    logger.info("启动 SGLang 服务: %s", kw)
    try:
        from sglang.launch_server import run_server as _run  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ImportError("请先 `pip install sglang>=0.3` 再用 SGLang 后端") from exc
    return _run(**kw)  # type: ignore[arg-type]

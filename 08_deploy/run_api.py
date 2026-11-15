"""OpenAI 兼容 API 启动入口（薄壳）。

用法::

    # CPU smoke：用手写后端 + 极小模型配置跑通（权重不存在时退化为随机权重）
    python 08_deploy/run_api.py --config configs/deploy/api.yaml \\
        --override model.device=cpu --override model.model_path=

    # 8B 真机
    python 08_deploy/run_api.py --config configs/deploy/api.yaml \\
        --override model.model_path=outputs/llama3_1_8b_pretrained

启动后即可访问：
- ``GET  /health``
- ``GET  /v1/models``
- ``POST /v1/chat/completions``（支持 ``stream: true`` 的 SSE）

注意：uvicorn / torch / 模型 / 分词器全部在函数内部惰性导入，本文件 import 阶段
不触碰任何重依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 三行 shim：保证零安装也能 import 到 src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from llama31_8b_scratch.common.config import load_config  # noqa: E402
from llama31_8b_scratch.common.logging_utils import get_logger, setup_logging  # noqa: E402
from llama31_8b_scratch.deployment.service import BaseModelService  # noqa: E402

logger = get_logger("run_api")


def _build_scratch_service(cfg) -> BaseModelService:
    """按配置构造手写后端服务（torch / 模型 / 分词器全部惰性导入）。"""
    import torch  # noqa: PLC0415
    from llama31_8b_scratch.deployment.service import ScratchCausalLMService  # noqa: PLC0415

    m = cfg.model
    device = str(getattr(m, "device", "cpu"))

    # 模型配置：复用 configs/model 下冻结 yaml
    from llama31_8b_scratch.common.config import Config  # noqa: PLC0415
    from llama31_8b_scratch.model.config import ModelConfig  # noqa: PLC0415

    model_cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "smoke.yaml"
    model_config = ModelConfig.from_yaml_config(Config.from_yaml(model_cfg_path))

    from llama31_8b_scratch.model.causal_lm import ScratchForCausalLM  # noqa: PLC0415

    model = ScratchForCausalLM(model_config)
    weight_path = str(getattr(m, "model_path", "") or "")
    if weight_path and Path(weight_path).exists():
        model.load_weights(weight_path)
        logger.info("已加载权重: %s", weight_path)
    else:
        logger.warning("model_path 为空或不存在，使用随机权重（仅 smoke 用）")
    model.to(device).eval()

    # 分词器：惰性用 tokenizers 库加载 HF 格式 tokenizer.json
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok_path = (
        Path(weight_path) / "tokenizer.json" if weight_path else None
    )
    if tok_path is not None and tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        logger.warning("未找到 tokenizer.json，用占位分词器（字符级，仅 smoke）")
        tokenizer = _CharTokenizer()

    return ScratchCausalLMService(model, tokenizer, model_name=str(m.model_name))


class _CharTokenizer:
    """smoke 用最小分词器：逐字符编码/解码，避免真分词器依赖。"""

    def encode(self, text: str):  # noqa: D102
        return [ord(c) % 512 for c in text]

    def decode(self, ids) -> str:  # noqa: D102
        return "".join(chr(i) for i in ids)


def build_service(cfg) -> BaseModelService:
    """按 backend 字段分发到对应后端。"""
    backend = str(getattr(cfg, "backend", "scratch"))
    if backend == "scratch":
        return _build_scratch_service(cfg)
    if backend in ("vllm", "sglang"):
        raise ValueError(
            f"backend={backend} 请用 08_deploy 直接启动对应引擎服务"
            f"（configs/deploy/{backend}.yaml），不要经本入口"
        )
    raise ValueError(f"未知 backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="llama31-8b-scratch OpenAI 兼容 API")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument(
        "--override", action="append", default=[], help="a.b=value，可重复"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)

    service = build_service(cfg)

    # uvicorn 惰性导入
    import uvicorn  # noqa: PLC0415

    from llama31_8b_scratch.deployment.api_server import create_app  # noqa: PLC0415

    app = create_app(service)
    server = cfg.server
    logger.info("启动 OpenAI 兼容服务于 %s:%s", server.host, server.port)
    uvicorn.run(app, host=server.host, port=int(server.port))


if __name__ == "__main__":
    main()

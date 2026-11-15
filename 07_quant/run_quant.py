"""量化入口：薄壳，按配置分发到手写 core / GPTQ / AWQ / bnb / GGUF。

用法::

    python 07_quant/run_quant.py --config configs/quant/smoke.yaml
    python 07_quant/run_quant.py --config configs/quant/llama3_1_8b.yaml \\
        --override mode=gptq
"""

from __future__ import annotations

import sys
from pathlib import Path

# 三行 shim
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from llama31_8b_scratch.common.config import load_config  # noqa: E402
from llama31_8b_scratch.common.logging_utils import get_logger, setup_logging  # noqa: E402
from llama31_8b_scratch.common.seed import set_seed  # noqa: E402
from llama31_8b_scratch.quantization import core  # noqa: E402

logger = get_logger("quant")


def _run_core(cfg) -> None:
    """手写数学 sanity check：随机张量走 int8/int4 往返，打印误差。"""
    shape = tuple(cfg.core.tensor_shape)
    rng = np.random.default_rng(cfg.seed)
    x = rng.standard_normal(shape)
    for bits in cfg.core.num_bits_list:
        for kind, fn in (
            ("symmetric", core.symmetric_quantize),
            ("affine", core.affine_quantize),
        ):
            r = fn(x, num_bits=bits)
            xh = r.dequantize()
            err = core.quantization_error(x, xh)
            bound = r.scale / 2
            logger.info(
                "[core] %s int%d: scale=%.4f z=%d max_err=%.4f (bound %.4f) "
                "rel_l2=%.4f",
                kind, bits, r.scale, r.zero_point, err["max_abs"], bound,
                err["relative_l2"],
            )


def _run_gptq(cfg) -> None:
    from llama31_8b_scratch.quantization.gptq import gptq_quantize

    g = cfg.gptq
    if not g.enabled:
        logger.info("gptq.enabled=false，跳过")
        return
    texts = _read_calibration(g.calibration_file)
    gptq_quantize(
        model_path=g.model_path,
        output_dir=g.output_dir,
        calibration_texts=texts,
        bits=g.bits,
        group_size=g.group_size,
        desc_act=g.desc_act,
    )


def _run_awq(cfg) -> None:
    from llama31_8b_scratch.quantization.awq import awq_quantize

    a = cfg.awq
    if not a.enabled:
        logger.info("awq.enabled=false，跳过")
        return
    texts = _read_calibration(a.calibration_file)
    awq_quantize(
        model_path=a.model_path,
        output_dir=a.output_dir,
        calibration_texts=texts,
        bits=a.bits,
        group_size=a.group_size,
    )


def _run_bnb(cfg) -> None:
    from llama31_8b_scratch.quantization.bnb import load_quantized_model

    b = cfg.bnb
    if not b.enabled:
        logger.info("bnb.enabled=false，跳过")
        return
    # 只演示加载，不做训练；真要用 QLoRA 请在 04_sft / 05_align 里接 PEFT
    load_quantized_model(
        b.model_path, mode=b.mode, compute_dtype=b.compute_dtype
    )


def _run_gguf(cfg) -> None:
    from llama31_8b_scratch.quantization.gguf import run_gguf_pipeline

    g = cfg.gguf
    if not g.enabled:
        logger.info("gguf.enabled=false，跳过")
        return
    cmds = run_gguf_pipeline(
        hf_model_dir=g.model_path,
        output_dir=g.output_dir,
        llama_cpp_dir=g.llama_cpp_dir,
        quant_type=g.quant_type,
        run=False,   # 先打印命令，人工确认后再把 run 改 True
    )
    logger.info("convert cmd: %s", " ".join(cmds.convert))
    logger.info("quantize cmd: %s", " ".join(cmds.quantize))


def _read_calibration(path: str) -> list[str]:
    """从 jsonl 读校准文本（每行 {"text": "..."}）。"""
    import json

    p = Path(path)
    if not p.is_file():
        logger.warning("校准文件不存在: %s，用空列表", path)
        return []
    texts = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="llama31-8b-scratch 量化入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)

    # 无论哪种工业路径，先跑手写 core 的 sanity check
    _run_core(cfg)

    mode = cfg.mode
    logger.info("量化模式: %s", mode)
    if mode == "core":
        return
    if mode == "gptq":
        _run_gptq(cfg)
    elif mode == "awq":
        _run_awq(cfg)
    elif mode == "bnb":
        _run_bnb(cfg)
    elif mode == "gguf":
        _run_gguf(cfg)
    else:
        raise ValueError(f"未知 mode: {mode}")


if __name__ == "__main__":
    main()

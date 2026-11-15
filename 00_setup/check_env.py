"""环境自检脚本：检查 Python 版本、核心依赖、可选重依赖与 CUDA 状态。

用法::

    python3 00_setup/check_env.py

脚本只做导入探测与信息打印，不安装任何包；核心依赖缺失时给出安装提示，
可选依赖缺失仅标记为未安装，不影响默认教学链路。
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# 零安装 shim
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (模块名, 导入名, 是否核心依赖)
CORE_PACKAGES = [
    ("torch", "torch", True),
    ("numpy", "numpy", True),
    ("yaml", "PyYAML", True),
    ("safetensors", "safetensors", True),
    ("tokenizers", "tokenizers", True),
]

OPTIONAL_PACKAGES = [
    ("transformers", "transformers"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("trl", "trl"),
    ("deepspeed", "deepspeed"),
    ("bitsandbytes", "bitsandbytes"),
    ("vllm", "vllm"),
    ("gradio", "gradio"),
]


def _probe(import_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "未安装"


def main() -> None:
    print("=" * 60)
    print("llama31-8b-scratch 环境自检")
    print("=" * 60)
    print(f"Python 版本 : {platform.python_version()} ({sys.executable})")
    print(f"操作系统   : {platform.system()} {platform.release()}")

    print("\n[核心依赖]")
    core_ok = True
    for import_name, pkg_name, required in CORE_PACKAGES:
        ok, version = _probe(import_name)
        flag = "OK  " if ok else "MISS"
        if not ok and required:
            core_ok = False
        print(f"  {flag} {pkg_name:<14} {version}")

    print("\n[可选重依赖]（缺失不影响默认教学链路）")
    for import_name, pkg_name in OPTIONAL_PACKAGES:
        ok, version = _probe(import_name)
        flag = "OK  " if ok else "--  "
        print(f"  {flag} {pkg_name:<14} {version}")

    # torch / CUDA 状态
    try:
        import torch

        print("\n[torch / 设备]")
        print(f"  torch 版本   : {torch.__version__}")
        print(f"  CUDA 可用   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 设备数  : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\n[torch / 设备] torch 未安装，跳过设备检测")

    print("\n" + "=" * 60)
    if core_ok:
        print("结论：核心依赖齐备，可运行 CPU 冒烟链路。")
    else:
        print("结论：存在核心依赖缺失，请先按 00_setup/README.md 安装。")
    print("=" * 60)


if __name__ == "__main__":
    main()

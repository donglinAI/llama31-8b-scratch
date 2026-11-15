"""模型模块自检入口：前向 → 落盘 → 回填 → 生成。

用法（在项目根目录执行）::

    python3 02_model/run_model_check.py --config configs/model/smoke.yaml
    python3 02_model/run_model_check.py --config configs/model/smoke.yaml \
        --override model.num_hidden_layers=4 --max-new-tokens=12

本脚本是薄入口：实质实现全部位于 ``src/llama31_8b_scratch/model`` 与
``src/llama31_8b_scratch/generation``，保证可被单元测试直接 import。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import tempfile

import torch

from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.common.seed import set_seed
from llama31_8b_scratch.generation import generate
from llama31_8b_scratch.model.causal_lm import ScratchForCausalLM
from llama31_8b_scratch.model.config import ModelConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数：``--config`` 必填，``--override`` 可重复。"""
    parser = argparse.ArgumentParser(
        description="手写 Transformer 模型自检：前向、落盘、回填、生成"
    )
    parser.add_argument("--config", required=True, help="模型 YAML 配置路径")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="点号路径覆盖项，形如 model.hidden_size=64，可重复传入",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=8, help="自检生成的最大新 token 数"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    # 1) 加载配置并构建模型
    raw_config = load_config(args.config, args.override)
    model_config = ModelConfig.from_yaml_config(raw_config)
    model = ScratchForCausalLM(model_config)
    model.eval()
    print(f"[1] 模型构建完成，参数量: {model.num_parameters():,}")

    # 2) 前向 + 损失（随机 prompt，仅验证形状与数值有限）
    prompt = torch.randint(0, model_config.vocab_size, (1, 4))
    forward_out = model(prompt, labels=prompt)
    print(
        f"[2] 前向 logits 形状: {tuple(forward_out.logits.shape)}, "
        f"loss: {float(forward_out.loss):.4f}"
    )

    # 3) 落盘 safetensors + config.json，再回填到一个全新模型
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        print(f"[3] 权重与 config.json 已写入临时目录: {tmp_dir}")

        reloaded = ScratchForCausalLM(model_config)
        reloaded.load_weights(tmp_dir)
        reloaded.eval()

        # 4) 回填后做一次贪心生成，验证权重可用
        generated = generate(
            reloaded,
            prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
            eos_token_id=model_config.eos_token_id,
            use_cache=True,
        )
        print(f"[4] 回填后生成序列形状: {tuple(generated.shape)}")

    print("自检完成：模型前向、落盘、回填、生成链路全部通过。")


if __name__ == "__main__":
    main()

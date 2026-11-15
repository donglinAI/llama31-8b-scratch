"""llama31_8b_scratch：从零手写的工业级 Llama3.1 全流程实战包。

模块划分与项目搭建顺序一一对应：

- :mod:`llama31_8b_scratch.common`       公共基础设施（配置、日志、种子、路径）
- :mod:`llama31_8b_scratch.model`        手写 Transformer 解码器（RMSNorm/RoPE/GQA/SwiGLU）
- :mod:`llama31_8b_scratch.generation`   生成与解码策略、KV Cache
- :mod:`llama31_8b_scratch.data`         分词器训练、数据处理、数据集与 collator
- :mod:`llama31_8b_scratch.training`     预训练 / 指令微调训练循环与工业级训练封装
- :mod:`llama31_8b_scratch.align`        奖励模型、PPO、DPO、ORPO、GRPO 对齐算法
- :mod:`llama31_8b_scratch.evaluation`   困惑度、多选评测与 lm-eval 封装
- :mod:`llama31_8b_scratch.quantization` 手写量化数学与 GPTQ/AWQ/bnb/GGUF 工业量化
- :mod:`llama31_8b_scratch.deployment`   OpenAI 兼容服务、vLLM/SGLang、Docker
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

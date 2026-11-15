# 00 环境搭建：依赖分层、CUDA 与常见安装坑

## 1. 本期解决什么问题

本期位于全链路最前置位置：在动手写模型、训练与对齐代码之前，先把 Python 环境、
核心依赖与可选重依赖的边界讲清楚。输入是一台干净机器（或一个新建虚拟环境），
输出是一套「最小依赖即可跑通 CPU 冒烟链路、按需再装重依赖」的可复现环境。

llama31-8b-scratch 把依赖分成两层：

- **核心依赖**（`requirements.txt` 基础集合）：torch、numpy、PyYAML、tqdm、
  safetensors、tokenizers。只装这一层即可导入全部手写模块并跑通单测；
- **可选重依赖**（train/quant/deploy/eval 各分组）：datasets、accelerate、peft、
  trl、deepspeed、vllm、gradio 等。代码内部一律惰性导入，缺失时仅对应功能不可用，
  不会让 `import llama31_8b_scratch` 直接报错。

## 2. 核心原理

### 2.1 为什么要做依赖分层

大模型项目的重依赖（如 vLLM、DeepSpeed、bitsandbytes）对 CUDA 版本、编译器和
内核耦合很强，在 CPU 机器或新驱动环境里经常装不上。教学代码的首要目标是可读、
可跑，因此把「必须能导入」的核心逻辑与「规模化才需要」的库解耦：

- 纯算法逻辑（模板渲染、清洗去重、量化数学、优势归一化）只依赖标准库 / numpy；
- 数据集、模型结构依赖 torch（核心依赖）；
- 训练框架、推理引擎、量化后端全部惰性导入，在函数/方法内部 `import`。

这样在只有 CPU 的环境里，依然可以完成分词、数据管道、模型前向与单测验证。

### 2.2 依赖体积与安装顺序

建议顺序：先建虚拟环境 → 装核心依赖 → 跑环境自检 → 进入后续阶段再按需补重依赖。
避免一上来 `pip install -r requirements.txt` 把所有重依赖拉入，导致 CPU 机器在
bitsandbytes / vLLM 等包上编译失败。

## 3. 关键代码精读

环境自检脚本：`00_setup/check_env.py`，核心探测逻辑如下：

```python
def _probe(import_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "未安装"
```

- 核心依赖表 `CORE_PACKAGES` 任一缺失会在结论里提示需要安装；
- 可选依赖表 `OPTIONAL_PACKAGES` 只打印状态，不阻断；
- torch 单独探测 CUDA：`torch.cuda.is_available()` 与设备名列表，便于判断
  当前 wheel 是否带 CUDA 支持。

包内部的惰性导入约定示例（以 SFT 数据集为例，重依赖只在用到时才进函数体）：

```python
def load_with_datasets(path: str | Path, split: str | None = None):
    try:
        from datasets import load_dataset  # 函数内部惰性导入
    except ImportError as exc:
        raise ImportError("未安装 datasets；标准 jsonl 流程请使用 read_jsonl") from exc
    return load_dataset("json", data_files=str(path), split=split or "train")
```

## 4. 运行命令

### 4.1 创建环境并安装核心依赖

```bash
# 建议 Python 3.10+
python3 -m venv .venv
source .venv/bin/activate

# 核心依赖（CPU 版 torch 示例；有 GPU 时按 PyTorch 官网选择 cu 构建）
pip install torch numpy PyYAML tqdm safetensors tokenizers

# 环境自检
python3 00_setup/check_env.py
```

### 4.2 源码可编辑安装（可选）

```bash
pip install -e .
```

### 4.3 真机 GPU 环境（8B 训练/推理）

```bash
# 先确认驱动与 CUDA 版本
nvidia-smi

# 按 PyTorch 官网指令安装对应 cu 构建的 torch，例如 cu121
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 训练阶段再装训练重依赖
pip install datasets accelerate peft trl deepspeed
```

### 4.4 运行项目单测（CPU）

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 5. 常见坑与排查

1. **`torch.cuda.is_available()` 返回 False，但驱动正常**
   - 现象：`nvidia-smi` 可见显卡，但 Python 里 CUDA 不可用。
   - 排查：多半装了 CPU 版 torch，或 torch 构建的 CUDA 版本高于驱动支持。
     用 `python3 -c "import torch; print(torch.version.cuda)"` 查看构建版本，
     对照 `nvidia-smi` 右上角的 CUDA 版本，必要时重装对应 cu 构建。

2. **`ImportError: libcuda.so.1: cannot open shared object file`**
   - 现象：在无显卡容器里安装了带 CUDA 的 torch 变体，加载时报共享库缺失。
   - 排查：CPU 机器改装 CPU 版 torch；不要在纯 CPU 环境强行安装 GPU 版 wheel。

3. **bitsandbytes / deepspeed 编译失败**
   - 现象：`pip install bitsandbytes` 报 C++ 编译错误，或 deepspeed 检查器报
     CUDA 编译器缺失。
   - 排查：这两个是训练量化重依赖，CPU 教学链路不需要装；先确认是否真的要走
     8bit/4bit 或 ZeRO 训练，再按其官方文档准备 CUDA 工具链，不要在干净环境里
     顺带安装。

4. **`ModuleNotFoundError: No module named 'llama31_8b_scratch'`**
   - 现象：直接运行编号目录脚本时找不到包。
   - 排查：脚本顶部三行 shim 会把 `src/` 注入 `sys.path`；若从其他目录调用，
     先执行 `export PYTHONPATH=src`，或使用 `pip install -e .`。

5. **版本冲突导致 numpy/torch ABI 报错**
   - 现象：`numpy.ndarray is not a numpy array` 或 torch 扩展加载崩溃。
   - 排查：保持 numpy 1.24+ 与 torch 官方配套范围；建议新建干净虚拟环境，
     避免与系统全局包混装。

## 下一期

- [01 分词器：BPE 训练、对话模板与词表扩充](../01_tokenizer/README.md)

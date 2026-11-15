# 07 量化：从手写对称量化到 GPTQ / AWQ / NF4 / GGUF

## 1. 本期解决什么问题

8B 模型 fp16 要占 16GB 显存，4bit 只要 ~5GB；CPU/边缘端跑 7B 模型，
GGUF Q4_K_M 只要 ~4GB。量化让同一个模型在更小的显存/内存里跑起来，代价是
精度略降。本期在全链路里处于 **评测之后、部署之前**：先用量化数学讲清楚
"压精度到底压了什么"，再讲四种工业路线的取舍。

- **输入**：HF 格式模型目录（或合成张量做 sanity check）。
- **输出**：
  - 手写 core：随机张量 int8/int4 往返的误差统计；
  - GPTQ：离线 4bit 权重 checkpoint；
  - AWQ：激活感知 4bit checkpoint；
  - bnb：加载时 4bit NF4（QLoRA 用）；
  - GGUF：llama.cpp 可直接跑的 `.gguf` 文件。

## 2. 核心原理

### 2.1 均匀量化的三步数学

设浮点张量 $x$ 映射到 $b$ 位有符号整数网格，$q_{\min}=-2^{b-1}$，
$q_{\max}=2^{b-1}-1$（int8: [-128,127]，int4: [-8,7]）。

**仿射（非对称）量化**：

$$s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad
z = \mathrm{round}\!\left(q_{\min}-\frac{x_{\min}}{s}\right)$$

$$q = \mathrm{clip}\!\left(\mathrm{round}(x/s)+z,\ q_{\min}, q_{\max}\right),\quad
\hat x = s(q-z)$$

**对称量化**（zero_point 强制为 0）：

$$m = \max(|x_{\min}|,|x_{\max}|),\quad s = m/q_{\max},\quad
q = \mathrm{clip}(\mathrm{round}(x/s), q_{\min}, q_{\max})$$

**误差边界**：均匀量化的最大逐元素误差不超过半个网格步长 $s/2$。
int4 的 scale 比 int8 大约 16 倍，误差也大约 16 倍——这就是为什么
4bit 必须用 group size / 激活感知 / 二阶补偿来把误差压回去。

### 2.2 GPTQ：用 Hessian 补偿量化误差

对某线性层 $W\in\mathbb R^{o\times d}$，用校准集收集输入激活
$X\in\mathbb R^{n\times d}$，构造 Hessian：

$$H = X^\top X$$

按列 $c=1,\dots,d$ 推进：量化第 $c$ 列产生残差 $\delta_c = W[:,c]-q_c$，
立刻摊到后面还没量化的列：

$$W[:,j] \mathrel{+}= -\frac{\delta_c}{H_{cc}}H_{cj},\quad j>c$$

数值上用 Cholesky $H=LL^\top$ 加正则 $\epsilon I$ 实现。这就是为什么 GPTQ
只需要几百条校准文本就能把 4bit 误差压到接近 fp16。

### 2.3 AWQ：激活感知权重缩放

按输入通道 $j$ 选可逆缩放：

$$w'_j = s_j w_j,\quad x'_j = x_j/s_j$$

数学上输出不变。$s_j$ 按该通道平均激活幅度 $\bar a_j$ 与平均权重幅度
$\bar w_j$ 选：

$$s_j = \bar a_j^{\alpha}/\bar w_j^{1-\alpha},\quad \alpha\approx 0.5$$

把"激活大、权重小"的关键通道在量化网格上放大，不浪费精度。

### 2.4 bnb NF4：非均匀网格匹配正态分布

NF4 的 16 个量化级按标准正态分位点布置：

$$q_i = \Phi^{-1}(i/15),\quad i=0,\dots,15$$

因为观察到归一化后的权重近似正态，0 附近权重大头需要精细网格。
**double quant** 把 fp32 scale 再量化成 int8，省 ~0.5 bit/参数。
**paged optimizer** 是 QLoRA 训练时把 AdamW state 分页 offload 到 CPU。

### 2.5 GGUF：llama.cpp 的块量化

K-quants 把权重切成小块，块内用不同 scale，outlier 通道单独保留更高精度。
`Q4_K_M` 是 4.8 bit/权重，体积/质量最平衡。

## 3. 关键代码精读

| 文件 | 作用 |
|---|---|
| `core.py` | `symmetric_quantize` / `affine_quantize` / `QuantResult.dequantize` / `quantization_error` |
| `gptq.py` | `gptq_quantize`：惰性 `auto_gptq`，Hessian/Cholesky 原理写在 docstring |
| `awq.py` | `awq_quantize`：惰性 `autoawq`，权重缩放公式写在 docstring |
| `bnb.py` | `build_4bit_config` / `build_8bit_config` / `load_quantized_model` |
| `gguf.py` | `build_convert_command` / `build_quantize_command` / `run_gguf_pipeline` |

手写 core 的关键片段（`core.py`）：

```python
scale = (x_max - x_min) / (q_max - q_min)
zero_point = int(round(q_min - x_min / scale))
q = np.clip(np.round(x / scale) + zero_point, q_min, q_max)
x_hat = scale * (q - zero_point)
```

## 4. 运行命令

### CPU smoke（合成张量，不调工业库）

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch
PYTHONPATH=src python3 07_quant/run_quant.py --config configs/quant/smoke.yaml
```

预期看到 int8 相对 L2 误差 ~1%、int4 ~20%——这是 per-tensor 朴素量化的
基线，工业库就是把这个数字压下去。

### 8B 真机：GPTQ 4bit

```bash
pip install -e ".[quant]"
PYTHONPATH=src python3 07_quant/run_quant.py \
    --config configs/quant/llama3_1_8b.yaml \
    --override mode=gptq \
    --override gptq.model_path=outputs/llama3_1_8b_pretrained
```

### 8B 真机：GGUF Q4_K_M（需要本地 llama.cpp 仓库）

```bash
git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && cmake -B build && cmake --build build -j

PYTHONPATH=src python3 07_quant/run_quant.py \
    --config configs/quant/llama3_1_8b.yaml \
    --override mode=gguf \
    --override gguf.enabled=true \
    --override gguf.llama_cpp_dir=/opt/llama.cpp
```

默认 `run=False` 只打印命令；确认无误后把 `run_gguf_pipeline(..., run=True)`
打开。

### 单测

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "quant_*.py" -v
```

## 5. 常见坑与排查

1. **int4 误差 20% 是正常的**：per-tensor 朴素量化就是这个水平。看到
   误差从 20% 降到 2%~3% 才是 GPTQ/AWQ 在起作用，不要以为 core 写错了。
2. **`auto_gptq` 装不上**：它对 torch/CUDA 版本很挑，建议用它官方指定的
   torch 组合；CPU 环境跑不了 GPTQ 校准（需要 GPU）。
3. **GGUF 转换报 `KeyError: 'llama'`**：你的 HF 模型 `config.json` 里
   `model_type` 不是 `llama`（或我们手写模型没保存对）。本项目手写模型
   保存时已经写 `model_type="llama"`，直接转即可。
4. **bnb 4bit 在 CPU 报 CUDA error**：bitsandbytes 本质是 CUDA 库，
   CPU 上不能用。CPU 量化走 GGUF；GPU 上 QLoRA 才用 bnb。
5. **zero_point 越界**：当 x_min 非常接近 0 时，round 出来的 zero_point
   可能超出 [q_min, q_max]，必须 clip（我们的 `core.py` 已经 clip）。

## 下一期

[08 部署](../08_deploy/README.md)：把量化好的模型用 vLLM/SGLang 包成
OpenAI 兼容 API，配上 Docker。

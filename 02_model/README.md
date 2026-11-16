# 02 手写 Transformer 模型与生成解码

> 本期对应代码：`src/llama31_8b_scratch/model/`、`src/llama31_8b_scratch/generation/`；自检入口：`02_model/run_model_check.py`；单元测试：`tests/core/model_*.py`。

## 1. 本期解决什么问题

在全链路中，分词器把文本变成 token id 之后，本期负责把一串 token id 映射为「下一个词元的概率分布」，并在此基础上完成自回归生成与解码策略。它向上承接数据/分词模块，向下支撑预训练、指令微调与对齐阶段。

| 方向 | 内容 |
|---|---|
| 输入 | `input_ids`: `[B, T]` 词元下标；可选 `labels`: `[B, T]` 标签 |
| 模型内部 | 嵌入查表 → N 层 pre-norm 解码器块 → 最终 RMSNorm → 词表投影头 |
| 输出 | `logits`: `[B, T, V]`；训练时给出移位交叉熵损失；推理时逐步采样新 token |
| 落盘兼容 | state_dict 键名与 HuggingFace Llama 逐字一致，可被 transformers / vLLM / SGLang 以 `model_type="llama"` 直接加载 |

本期不做的事：分词器训练（见 `01_tokenizer/`）、训练循环（见 `03_pretrain/`）、奖励对齐（见 `05_align/`）、量化与服务部署（见 `07_quant/`、`08_deploy/`）。

## 2. 核心原理

### 2.1 整体结构：pre-norm 解码器栈

Llama 系列采用 **pre-norm** 残差结构。每个解码器块包含两个子层：自注意力与前馈网络，子层输入先做 RMSNorm 再进入变换，输出与残差相加：

$$
x' = x + \mathrm{Attn}\big(\mathrm{RMSNorm}(x)\big)
$$

$$
x'' = x' + \mathrm{MLP}\big(\mathrm{RMSNorm}(x')\big)
$$

堆叠 $N$ 层后接最终归一化与词表投影。张量形状贯穿全程：

- 嵌入输出：`[B, T, H]`（`B` 批大小，`T` 序列长度，`H` 隐藏维）；
- 注意力内部：Q `[B, n_h, T, D]`，K/V `[B, n_{kv}, T, D]`，`D = head_dim`；
- 投影头输出：`[B, T, V]`，`V` 为词表大小。

### 2.2 RMSNorm

对长度为 $d$ 的向量 $x \in \mathbb{R}^{d}$，均方根为：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^{2} + \epsilon}
$$

逐维缩放并施加可学习权重 $w$：

$$
\hat{x}_i = \frac{x_i}{\mathrm{RMS}(x)} \cdot w_i
$$

与 LayerNorm 相比，RMSNorm 不减均值、无偏置项，计算量更小。实现时方差在 fp32 下计算，避免半精度时 $x^2$ 下溢导致归一化偏差。

### 2.3 旋转位置编码（RoPE）

设单头维度 $D = \mathrm{head\_dim}$，位置 $t$，基频 $\theta_{\text{base}}$。对第 $i$ 个成对维度（$i = 0, 1, \dots, D/2 - 1$）定义角频率：

$$
\omega_i = \theta_{\text{base}}^{-\frac{2i}{D}}
$$

位置 $t$ 的相位为 $t\omega_i$。RoPE 把成对维度 $(x_{2i}, x_{2i+1})$ 当作二维向量在其平面上旋转：

```math
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}=\begin{pmatrix} \cos(t\omega_i) & -\sin(t\omega_i) \\ \sin(t\omega_i) & \cos(t\omega_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
```

等价的张量写法使用 `rotate_half`：把最后一维对半拆成 $x_1 = x_{:D/2}$ 、$x_2 = x_{D/2:}$ ，则

$$
\mathrm{rotate\_half}(x) = [ -x_2,\ x_1 ]
$$

于是旋转可写成逐元素运算：

$$
x' = x \odot \cos(\mathrm{emb}) + \mathrm{rotate\_half}(x) \odot \sin(\mathrm{emb})
$$

**长上下文 scaling**：

- **linear（线性插值）**：频率整体除以缩放因子 $s$，相当于把位置轴拉长 $s$ 倍：$\omega_i \leftarrow \omega_i / s$。
- **dynamic（NTK-by-parts）**：提高 RoPE 基频，使高频维度（精确位置）基本不变、低频维度（粗粒度结构）被插值。等价实现：

  $$
  \omega_i' = \omega_i \cdot s^{\frac{2i}{D}}
  $$

- **YaRN（Yet Another RoPE Extension）**：完整方法在 NTK 基础上引入「修正维度区间」与注意力幅度缩放（mscale）。先按旋转次数定位需要修正的低频/高频维度区间，在区间内做频率插值；再用

  $$
  \mathrm{mscale}(s) = 0.1 \cdot m \cdot \ln(s) + 1
  $$

  对注意力 logits 做幅度补偿，缓解长上下文注意力分布被稀释。本项目代码侧 `yarn` 分支先回退基础 RoPE 并打印 warning，完整数值实现列入后续迭代（TBD）；`linear` 与 `dynamic` 已完整手写。

### 2.4 注意力与 GQA

缩放点积注意力：

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{D}}\right)V
$$

**分组查询注意力（GQA）**：Q 头数 $n_h$，KV 头数 $n_{kv}$，分组大小 $g = n_h / n_{kv}$。每个 KV 头服务 $g$ 个 Q 头。手写 `repeat_kv` 在 KV 头维度复制到与 Q 头一一对应：

$$
K_{\mathrm{rep}}[b, h, t, d] = K[b, h // g, t, d]
$$

三种注意力实现可切换：

- `eager`：手写 $QK^\top/\sqrt{D}$ → 加因果 mask → softmax → 乘 V，教学路径；
- `sdpa`：`F.scaled_dot_product_attention`，底层自动选择高效内核；
- `flash_attention_2`：惰性导入 `flash_attn`；本机未安装时打印 warning 并回退 `sdpa`。FlashAttention-2 的核心是分块（tiling）计算、不落地完整注意力矩阵，从而把 HBM 读写量降一个量级；FlashAttention-3 进一步利用异步流水线与 FP8 数据类型，本期在文档层面讲解，代码不绑定特定 FA 版本。

### 2.5 SwiGLU 前馈网络

Llama 的 FFN 使用门控 SwiGLU 形式：

$$
\mathrm{SwiGLU}(x) = W_{\mathrm{down}}\big(\sigma(W_{\mathrm{gate}} x) \odot W_{\mathrm{up}} x\big)
$$

其中 $\sigma$ 为 SiLU（Swish）：$\sigma(z) = z \cdot \mathrm{sigmoid}(z)$。三个投影分别为 `gate_proj`、`up_proj`、`down_proj`，无 bias。

### 2.6 因果 mask 与标签移位损失

解码器只能看到当前位置及之前的 token，因此构造加性因果 mask：允许位置填 $0$，屏蔽位置填 $-\infty$，softmax 后对应权重趋零。

训练损失在模型内部完成移位：用位置 $t$ 的 logits 预测位置 $t+1$ 的标签：

```math
\begin{aligned}
\mathcal{L} &= -\sum_{t=1}^{T-1}\log P(y_{t+1}\mid y_{\le t};\theta) \\
&= -\Big[ \log P(y_{2}\mid y_1;\theta) +\log P(y_{3}\mid y_1,y_2;\theta) +\dots +\log P(y_{T}\mid y_1,\dots,y_{T-1};\theta) \Big]
\end{aligned}
```

对应代码：`cross_entropy(logits[..., :-1, :], labels[..., 1:])`，`ignore_index=-100` 的位置不计损失。

### 2.7 KV Cache 与解码复杂度

无缓存时，每生成一个新 token 都要把整段历史重新前向，注意力复杂度随长度平方增长。KV Cache 把每层历史 K/V 缓存下来，解码步只计算新 token 的 K/V 并追加到缓存尾部，每步注意力开销从 $O(T^2 D)$ 降到 $O(T D)$。缓存布局：`key_cache[layer]` / `value_cache[layer]` 均为 `[B, n_kv, T, D]`。

### 2.8 解码采样策略

- **Temperature**：$z_i' = z_i / T$。$T<1$ 分布更尖锐，$T>1$ 更平坦；$T=0$ 退化为贪心 argmax。
- **Top-k**：只保留概率最高的 $k$ 个候选，其余 logits 置 $-\infty$。
- **Top-p（核采样）**：按概率降序累计，只保留累计概率不超过 $p$ 的最小候选集合。
- **重复惩罚**：对已出现 token 调整 logits，降低重复概率：

$$
z_i' = \begin{cases} z_i / p, & z_i \ge 0 \\ z_i \cdot p, & z_i < 0 \end{cases}
$$

$p>1$ 抑制重复。

## 3. 关键代码精读

### 3.1 配置与 HF 兼容导出

`src/llama31_8b_scratch/model/config.py` 中的 `ModelConfig` 字段与 YAML `model` 节点一一对应；`to_hf_dict()` 产出可直接写入 `config.json` 的字典，`model_type="llama"`，并保留经典字段 `rope_theta`、`rope_scaling`，兼容 transformers 4.x / 5.x 与 vLLM、SGLang 的读取路径。

### 3.2 RMSNorm 实现

`src/llama31_8b_scratch/model/norm.py` 的 `ScratchRMSNorm`：

```python
variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

`rsqrt(variance + eps)` 即 $1/\mathrm{RMS}(x)$；先 fp32 算方差，再恢复原始 dtype。

### 3.3 RoPE 与 rotate_half

`src/llama31_8b_scratch/model/rope.py`：

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

`ScratchRotaryEmbedding.forward` 产出 `[B, 1, T, D]` 的 cos/sin，广播到多头。

### 3.4 GQA 与三种注意力内核

`src/llama31_8b_scratch/model/attention.py` 的 `repeat_kv`：

```python
hidden_states = hidden_states[:, :, None, :, :].expand(
    batch, num_kv_heads, n_rep, seq_len, head_dim
)
return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
```

前向中先投影成多头、施加 RoPE、更新缓存，再 `repeat_kv` 复制 KV 头，最后按 `eager / sdpa / flash_attention_2` 分派。FA2 路径在方法内部 `from flash_attn import flash_attn_func`，缺失时 warning 并回退 sdpa，保证纯 CPU 环境 import 不报错。

### 3.5 解码器层与梯度检查点

`src/llama31_8b_scratch/model/decoder.py` 的 `ScratchDecoderLayer`：两个残差子层按 2.1 节公式堆叠；`gradient_checkpointing=True` 且处于训练态时，用 `torch.utils.checkpoint.checkpoint` 包裹前向，反向时重算中间激活以节省显存。

### 3.6 因果 LM 与权重落盘

`src/llama31_8b_scratch/model/causal_lm.py` 的 `ScratchForCausalLM`：

- 前向返回 `logits` 与可选 `loss`；绑定词嵌入时复用 `embed_tokens.weight` 做投影；
- `save_pretrained` 写 `model.safetensors` 与标准 `config.json`；
- `load_weights` 回填权重，允许绑定权重导致的 `lm_head.weight` 缺失。

### 3.7 KV Cache 与生成循环

`src/llama31_8b_scratch/generation/cache.py` 的 `KVCache.update` 在层尾部按序列维度拼接 K/V；`src/llama31_8b_scratch/generation/generate.py` 的 `_run_decode` 先 prefill 整段 prompt，再逐步采样、追加、判定 eos，解码步配合缓存只喂新 token。`stream_generate` 是同一循环的流式版本，逐步产出 `token / sequence / step / finished`。

## 4. 运行命令

以下命令均在项目根目录 `llama31-8b-scratch/` 下执行。

### 4.1 运行模型单元测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "model_*.py" -v
```

### 4.2 CPU 冒烟自检（smoke 配置）

```bash
PYTHONPATH=src python3 02_model/run_model_check.py --config configs/model/smoke.yaml
```

带命令行覆盖项示例（把层数改为 4、生成 12 个新 token）：

```bash
PYTHONPATH=src python3 02_model/run_model_check.py \
    --config configs/model/smoke.yaml \
    --override model.num_hidden_layers=4 \
    --max-new-tokens 12
```

### 4.3 8B 真机配置校验

`configs/model/llama3.1_8b.yaml` 是 Llama 3.1 8B 结构（32 层、hidden 4096、32 Q 头、8 KV 头、head_dim 128、intermediate 14336、词表 128256、theta 500000）。当前环境为 CPU，不在此实例化 8B 权重；真机 GPU 环境下可用同一自检入口校验结构：

```bash
PYTHONPATH=src python3 02_model/run_model_check.py \
    --config configs/model/llama3.1_8b.yaml --max-new-tokens 16
```

> 预训练权重的下载、转换与加载属于部署/训练阶段范畴，本阶段不产出可评测的生成质量分数（TBD）。

## 5. 常见坑与排查

1. **下游加载报 missing / unexpected key**：state_dict 键名必须与 HF Llama 逐字一致，例如 `model.layers.0.self_attn.q_proj.weight`、`model.norm.weight`。若自定义了属性名或嵌套层级，vLLM/SGLang 会报权重不匹配。核对 SPEC 5.2 清单与 `causal_lm.py` 落盘逻辑。
2. **顶层 `import flash_attn` 导致纯 CPU 环境报错**：可选重依赖必须惰性导入（函数/方法内部 import）。本项目 FA2 缺失时回退 sdpa；若在代码顶层导入 `flash_attn`，连 `import llama31_8b_scratch.model` 都会直接失败。
3. **解码步位置编号错位**：有 KV Cache 时，每个新 token 的 `position_ids` 必须等于当前缓存序列长度（prefill 后为原 prompt 长度），不能继续从 0 编号；否则 RoPE 相位错误，生成质量下降甚至发散。
4. **GQA 头数不整除或隐藏维不匹配**：`num_attention_heads` 必须能被 `num_key_value_heads` 整除，且 `hidden_size = num_attention_heads × head_dim`。`ModelConfig.validate` 会提前报错，而不是在矩阵乘法处出现难以定位的形状异常。
5. **半精度下归一化方差下溢**：RMSNorm 内部若直接在 fp16/bf16 算 $x^2$，小值平方会下溢；实现统一先转 fp32 算方差，再恢复输入 dtype。
6. **Top-p 过滤后候选为空**：累计概率 mask 必须向后平移一位，并显式把第 0 个候选保留，否则极端分布下可能所有 logits 都被置 $-\infty$，softmax 得到 NaN。

## 下一期

- 上一期：`../01_tokenizer/README.md`（分词器训练与词表扩充）
- 下一期：`../03_pretrain/README.md`（预训练循环、数据打包与混合精度训练）

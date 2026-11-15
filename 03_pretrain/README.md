# 03 · 预训练：从 token 流到基座模型

## 1. 本期解决什么问题

在全链路里，预训练位于「分词器 → 预训练 → SFT → 对齐 → 评测 → 部署」的第二段。
输入是大规模纯文本（已被 tokenize 成 token id 流），输出是一个具备基础语言能力的
基座权重（与 HF Llama 兼容的 `state_dict`，可直接被 vLLM/SGLang 加载）。

本期交付：
- 手写训练循环（`src/llama31_8b_scratch/training/trainer.py`）；
- 闭式 cosine + warmup 调度器（`scheduler.py`）；
- 交叉熵 + 可选 z-loss（`losses.py`）；
- AdamW / AdamW8bit 优化器构造（`optimizer.py`）；
- DeepSpeed ZeRO 1/2/3 与 FSDP 工业封装（`distributed.py`）；
- 预训练入口 `run_pretrain.py` 与 smoke / 8B 两套配置。

## 2. 核心原理

### 2.1 全局 batch size 的拆解

设 `MBS` = 单卡 micro-batch、`GAS` = 梯度累积步数、`N` = GPU 数：

$$
\text{GLOBAL\_BS} = \text{MBS} \times \text{GAS} \times N
$$

调度器只面对"优化步"，每 `GAS` 次 forward 才 `optimizer.step()` 一次。

### 2.2 label 移位与 -100 掩码

位置 $i$ 的 logits 预测位置 $i+1$ 的 token：

$$
L = \text{CrossEntropy}\big(\text{logits}[:, :-1, :],\ \text{labels}[:, 1:]\big)
$$

`labels == -100` 的位置被 PyTorch 忽略，用于 padding 与 SFT 的 prompt mask。

### 2.3 cosine + warmup 闭式公式

设 $T$ 为总优化步、$W$ 为 warmup 步、$\eta_{\min}$ 为末端比例：

$$
\eta(t) =
\begin{cases}
\text{base\_lr} \cdot \dfrac{t}{W}, & t < W \\[4pt]
\text{base\_lr} \cdot \big[\eta_{\min} + (1-\eta_{\min})\cdot \tfrac{1}{2}(1+\cos \pi p)\big], & t \ge W
\end{cases}
\quad p = \dfrac{t-W}{T-W}
$$

端点：$p=0$ 时回到 peak（连续），$p=1$ 时降到 `base_lr * eta_min`。

### 2.4 ZeRO 各 stage 切什么

| Stage | 优化器状态 | 梯度 | 模型权重 | 适用 |
|---|---|---|---|---|
| ZeRO-1 | ✅ 分片 | ❌ | ❌ | 1B~7B 单节点 |
| ZeRO-2 | ✅ | ✅ 分片 | ❌ | 8B 甜点 |
| ZeRO-3 | ✅ | ✅ | ✅ 分片 | 70B+ / 多节点 |

FSDP 是 PyTorch 原生的 ZeRO-3 等价物，与 `torch.compile` / 梯度检查点配合更紧；
DeepSpeed 生态更成熟、HF 集成最好。

## 3. 关键代码精读

调度器（`scheduler.py`）：

```python
def lr_at(self, step: int, base_lr: float) -> float:
    w, t_total = self.warmup_steps, self.total_steps
    if w > 0 and step < w:
        return base_lr * (step / w)                  # warmup 线性
    progress = min(1.0, max(0.0, (step - w) / (t_total - w)))
    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * factor)
```

训练循环核心（`trainer.py`）——梯度累积：

```python
with self._autocast_context():
    out = self.model(**batch)
    loss = self._extract_loss(out) / cfg.gradient_accumulation_steps
loss.backward()
if accum % cfg.gradient_accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
    self.optimizer.step(); self.scheduler.step()
    self.optimizer.zero_grad(set_to_none=True)
```

## 4. 运行命令

CPU smoke（1~2 分钟）：

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch
PYTHONPATH=src python3 03_pretrain/run_pretrain.py \
    --config configs/pretrain/smoke.yaml
```

8B 真机（8 卡，ZeRO-2，bf16）：

```bash
torchrun --nproc_per_node=8 03_pretrain/run_pretrain.py \
    --config configs/pretrain/8b.yaml
```

## 5. 常见坑与排查

1. **loss 为 NaN**：多半是 fp16 下未开 GradScaler 或 lr 过大。8B 请用 `bf16`，
   不要用 `fp16`；若必须 fp16，需在 autocast 外包 `torch.cuda.amp.GradScaler`。
2. **OOM 在 4K 序列**：`gradient_checkpointing: true` 必须开，MBS 从 8 降到 2~4，
   配合 `gradient_accumulation_steps` 保持 global batch 不变。
3. **断点续训 loss 突然跳变**：检查 `scheduler.load_state_dict` 是否恢复了步数；
   本实现的 `save_checkpoint` 会同时存 model/optimizer/scheduler/global_step。
4. **8bit 优化器 CPU 报错**：bitsandbytes 只在 CUDA Linux 可用，CPU smoke 用 `adamw`。

下一期：[04_sft](../04_sft/README.md)。

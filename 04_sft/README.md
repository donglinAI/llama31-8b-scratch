# 04 · SFT：用指令数据把基座模型调成对话助手

## 1. 本期解决什么问题

预训练模型只会"续写"，不会"按指令回答"。SFT（Supervised Fine-Tuning）用
`{"messages": [{"role":...,"content":...}]}` 格式的指令对话数据，让模型学会：
遇到 user 输入时，生成一段 assistant 回答。

与预训练的两个关键差异：
1. **只对回答部分算 loss**：prompt（system/user）的 label 被置 `-100`，不产生梯度；
2. **通常用 LoRA/QLoRA**：冻结 8B 基座，只训 <1% 的适配器参数，单卡 24G 可微调 8B。

## 2. 核心原理

### 2.1 prompt mask（只学回答）

`SFTCollator` 把每条消息渲染成 Llama3.1 模板后，对 assistant 以外的 token 位置
把 label 置为 `-100`。最终仍走同一个 `causal_lm_loss`：

$$
L_{\text{SFT}} = \frac{\sum_{t\in\text{assistant}} \text{CE}(\text{logits}_t, y_{t+1})}{\#\text{assistant tokens}}
$$

这样模型不会被强迫去模仿 prompt 的分布。

### 2.2 LoRA 低秩分解

$$
W' = W_0 + BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll d
$$

只更新 $A,B$，$W_0$ 冻结；推理时可把 $BA$ 合并回 $W_0$，零延迟。

### 2.3 QLoRA / NF4

把 $W_0$ 用 4-bit 归一化浮点（NF4）量化冻结，$A,B$ 保持 bf16 训练。
33B 模型可在单张 24G 卡微调。NF4 是为正态分布权重设计的 4-bit 格式，比 INT4 更准。

### 2.4 序列打包（packing）

SFT 数据长短不一，padding 浪费算力。packing 把多条短样本拼到 `seq_len` 内，
用 attention mask 隔开；吞吐量通常提升 30%~80%。

## 3. 关键代码精读

LoRA 注入（`peft_utils.py`）——peft 缺失时的降级分支：

```python
def apply_lora(model, cfg):
    try:
        from peft import get_peft_model, LoraConfig as PeftLoraConfig
    except Exception as exc:
        logger.warning("peft 不可用，走 manual_freeze：%s", exc)
        for p in model.parameters():
            p.requires_grad = False
        return model, "manual_freeze"
    model = get_peft_model(model, PeftLoraConfig(...))
    return model, "peft"
```

SFT 入口（`run_sft.py`）——按配置选择 LoRA：

```python
model = build_model(cfg)
model = maybe_apply_peft(model, cfg)   # peft 缺失时降级为冻结 base
trainer = Trainer(model=model, train_loader=loader, config=tcfg)
trainer.train()
```

## 4. 运行命令

CPU smoke：

```bash
PYTHONPATH=src python3 04_sft/run_sft.py --config configs/sft/smoke.yaml
```

8B LoRA 真机（8 卡）：

```bash
torchrun --nproc_per_node=8 04_sft/run_sft.py --config configs/sft/8b.yaml
```

走 TRL 工业路径（需 `pip install 'llama31-8b-scratch[train]'`）：

```bash
python 04_sft/run_sft.py --config configs/sft/8b.yaml --backend trl
```

## 5. 常见坑与排查

1. **模型只会重复 prompt**：检查 `SFTCollator` 是否真的把 prompt label 置了 `-100`；
   最稳妥的办法是打印一个 batch 的 `labels` 张量，assistant 段以外应全是 -100。
2. **LoRA 训了但不收敛**：检查 `apply_lora` 返回的 backend 是 `peft` 还是
   `manual_freeze`；后者意味着 peft 没装上，base 参数全冻结、没有可训参数。
3. **QLoRA 单卡 OOM**：确认 `use_qlora: true` 且 base 模型用 NF4 加载；
   同时把 `gradient_checkpointing` 打开、`per_device_train_batch_size` 降到 1~2。
4. **wandb 报权限错**：`report_to: none` 先跑通，再 `wandb login` 后切回 `wandb`。

下一期：[05_align](../05_align/README.md)（RM / DPO / GRPO）。

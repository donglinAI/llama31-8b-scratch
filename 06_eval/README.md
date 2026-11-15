# 06 评测：困惑度、loglikelihood 多选与 lm-eval-harness

## 1. 本期解决什么问题

模型训练完、量化完，第一件事就是回答："它到底比上一版好还是差？" 这一期
在全链路里处于 **训练/对齐/量化之后、部署之前**，是给模型打分的"裁判"环节。

- **输入**：一个 HF 格式模型目录（或我们手写的 `ScratchForCausalLM`），若干
  评测任务的 YAML 配置。
- **输出**：
  - 手写分支：PPL、多选 accuracy、生成式任务的 mean/CI；
  - 工业分支：lm-eval-harness 在 MMLU / CMMLU / C-Eval / GPQA 等任务上
    的标准分数。
- **边界**：本目录**不预填任何分数**——真实数字必须在真机上跑完才填，
  严禁编造。

## 2. 核心原理

### 2.1 困惑度（PPL）的三步无跳步推导

设语料共 $T$ 个 token，模型在第 $t$ 个位置对词表 $\mathcal V$ 输出 logits
$z_t$，真实 token 为 $y_t$。

**第一步 Log-softmax（数值稳定）**：

$$\log p_t(v) = z_t(v) - \log\sum_{v'} \exp z_t(v')$$

**第二步交叉熵 / NLL**：

$$\mathrm{NLL} = -\frac{1}{T}\sum_{t=1}^{T}\log p_t(y_t)$$

**第三步困惑度**：

$$\mathrm{PPL} = \exp(\mathrm{NLL}) = \prod_{t=1}^{T} p_t(y_t)^{-1/T}$$

几个必须记住的结论：

- 模型完美预测时 NLL=0，PPL=1；
- 在大小为 $V$ 的词表上**随机猜**时，每个 token 概率 $1/V$，
  $\mathrm{NLL}=\log V$，**PPL=V**——这是随机基线；
- 跨文档聚合必须按 token 数加权，**不能**把每篇 PPL 取算术平均。

### 2.2 多选题的两种范式

**范式 A：loglikelihood（判别式，MMLU/CMMLU/C-Eval 默认）**

对每个选项 $c$，把 `context + c` 拼起来，只对 $c$ 里的 token 求条件概率之和：

$$s(c) = \sum_{t=c_1}^{c_{|c|}}\log p_\theta(t \mid \text{context}, t_{<t})$$

选 $\arg\max_c s(c)$。不生成、不采样、快且可复现。

**范式 B：gen（生成式，GPQA/数学题）**

让模型真的吐出一段文本，再用正则/规则/judge 抽取判定。方差大，但更接近真实使用。

### 2.3 置信区间

样本均值的 95% 置信区间（正态近似）：

$$\bar x \pm z_{0.975}\cdot \frac{s}{\sqrt n},\quad z_{0.975}\approx 1.95996$$

我们在 `metrics_core.py` 里用逆误差函数反解任意分位数，不依赖 scipy。

## 3. 关键代码精读

所有实质实现都在 `src/llama31_8b_scratch/evaluation/`：

| 文件 | 作用 |
|---|---|
| `metrics_core.py` | 均值、样本标准差、accuracy、置信区间（仅 numpy/stdlib） |
| `perplexity.py` | `log_softmax` / `cross_entropy` / `nll` / `perplexity` / `corpus_ppl` |
| `multiple_choice.py` | `sequence_logprob` / `mc_choice_logprobs` / `rank_choice` / `mc_accuracy` |
| `generation_eval.py` | `GenSample` + `exact_match` / `regex_match` + `GenerationEval` 框架 |
| `lm_eval_harness.py` | 惰性封装 `lm_eval.simple_evaluate`，含任务名速查表 |

PPL 的核心片段（`perplexity.py`）：

```python
logp = log_softmax(logits, axis=-1)              # [T, V]
token_logp = logp[np.arange(T), targets]          # [T]
ce = -token_logp.mean()
ppl = math.exp(ce)
```

工业封装的关键是**惰性 import**（`lm_eval_harness.py`）：

```python
def _require_lm_eval():
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise ImportError("请先 pip install lm-eval[api]>=0.4.4") from exc
```

这样只装 torch+numpy 时 `import llama31_8b_scratch.evaluation` 不会报错。

## 4. 运行命令

### CPU smoke（秒级，合成数据，无需 GPU/数据集）

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch
PYTHONPATH=src python3 06_eval/run_eval.py --config configs/eval/smoke.yaml
```

### 8B 真机（单卡 5090 / A100，跑标准 benchmark）

```bash
# 先装工业依赖
pip install -e ".[eval]"

PYTHONPATH=src python3 06_eval/run_eval.py \
    --config configs/eval/llama3_1_8b.yaml \
    --override lm_eval.model_path=outputs/llama3_1_8b_pretrained \
    --override lm_eval.device=cuda
```

结果落在 `outputs/eval_8b/`，**真实分数 TBD**（跑完再填）。

### 单测

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "eval_*.py" -v
```

## 5. 常见坑与排查

1. **`ppl=V` 卡死**：词表上模型完全没学到东西，logits 接近 0/均匀分布。
   先检查 `targets` 是否和 `logits` 对齐（移位了没有）。
2. **`lm_eval` 报任务名不存在**：不同版本任务名会漂移。跑之前先调
   `list_available_tasks(["gpqa"])` 在本机确认；常见的是 `gpqa_diamond`
   而不是 `gpqa`。
3. **跨文档 PPL 偏高**：你把每篇文档各自的 PPL 取了算术平均。必须用
   `corpus_ppl` 把所有 token 的 log p 拼起来再求平均。
4. **多选 accuracy 永远 1/n**：选项长度不一致时，长选项天然 sum log p 更
   小（更负）。loglikelihood 范式要按 token 数归一化或统一截断到同样长度，
   lm-eval 内部已经处理；自己手写时务必检查。

## 下一期

[07 量化](../07_quant/README.md)：把 8B 模型压到 4bit 还能保住几分？
GPTQ / AWQ / NF4 / GGUF 的原理与取舍。

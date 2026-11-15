# 05 对齐（Alignment）：RM / PPO / DPO / ORPO / GRPO

> 本期对应源码：`src/llama31_8b_scratch/align/`；配置：`configs/align/`；单测：
> `tests/core/align_*.py`。所有数学逻辑纯 PyTorch 手写，TRL 仅作惰性工业封装。

## 1. 本期解决什么问题

预训练 + SFT 之后，模型学会了“说人话”，但还不会“说人话里更讨喜、更正确的那
一句”。对齐阶段用**偏好数据**（prompt + chosen + rejected）或**可验证奖励**，
把模型行为往人类偏好 / 规则要求上校准。

- **在全链路中的位置**：`03 预训练 → 04 SFT → 05 对齐 → 06 评测 → 07 量化 → 08 部署`。
  对齐的输入是 SFT 产物（策略模型）与偏好 / 问答数据，输出是对齐后的策略权重。
- **输入 schema**（SPEC 5.3）：
  - 偏好对（RM / DPO / ORPO）：`{"prompt": ..., "chosen": ..., "rejected": ...}`；
  - GRPO：`{"prompt": [{"role":"user","content":...}], "answer": "标准答案"}`，
    由规则奖励函数打分。
- **输出**：对齐后的策略模型；RM 阶段额外产出可打分的奖励模型。

本期覆盖五条主流路线：经典 RLHF 的 **PPO**（Schulman et al., 2017）、免采样的
**DPO**（Rafailov et al., 2023）、单模型免参考的 **ORPO**（Hong & Lee, 2024）、
无价值网络的 **GRPO**（DeepSeekMath, 2024；DeepSeek-R1, 2025），并在文末简述
**SimPO / KTO**。

## 2. 核心原理

### 2.1 统一记号与张量形状

设策略模型为 :math:`\pi_\theta`，参考模型为 :math:`\pi_{\mathrm{ref}}`。对一条
回答 :math:`y=(y_1,\dots,y_T)`，逐位置对数概率由 logits 经 log-softmax 取得：

$$
\ell_{b,t}=\log\pi_\theta(y_t\mid y_{<t},x),\quad
\text{token logps 形状 }[B,T]
$$

序列对数概率（求和）：

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\ell_{b,t}\,m_{b,t},\quad
\text{序列 logps 形状 }[B]
$$

其中 :math:`m_{b,t}` 为有效位置掩码（prompt / padding 按数据集 collator 置 -100）。

### 2.2 奖励模型（RM）

RM 在 backbone 顶部接标量打分头 :math:`r(x,y)`，用 Bradley-Terry 成对损失训练：

$$
\mathcal{L}_{RM}=-\frac{1}{B}\sum_{i=1}^{B}\log\sigma\big(r(x_i,y_i^w)-r(x_i,y_i^l)\big)
$$

数值稳定写法直接用 `F.logsigmoid`，避免先 sigmoid 再取对数的饱和与下溢。

### 2.3 PPO（经典 RLHF）

四模型闭环：策略 :math:`\pi_\theta`、冻结参考 :math:`\pi_{\mathrm{ref}}`、奖励
模型 :math:`r`、价值头 :math:`V`。

**GAE 无跳步推导**（Schulman et al., 2016）：TD 残差

$$
\delta_t=r_t+\gamma V_{t+1}(1-d_t)-V_t
$$

广义优势

$$
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},\qquad R_t=A_t+V_t
$$

**裁剪策略目标**（:math:`\rho_t=\exp(\logp_\theta-\logp_{\mathrm{old}})`）：

$$
L^{CLIP}=-\mathbb{E}_t\big[\min(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)A_t)\big]
$$

**价值裁剪**：

$$
L^{VF}=\mathbb{E}_t\big[\max((V_\theta-R_t)^2,\ (V^{clipped}-R_t)^2)\big]
$$

**KL 惩罚**把策略拉住参考模型：

$$
L=L^{CLIP}+c_v L^{VF}+\beta_{KL}\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

### 2.4 DPO

DPO 把 KL 约束奖励最大化解析求解，得到隐式奖励
:math:`r(x,y)=\beta[\log\pi_\theta(y|x)-\log\pi_{\mathrm{ref}}(y|x)]`，成对损失：

$$
\mathcal{L}_{DPO}=-\frac{1}{B}\sum_i\log\sigma\Big(
\beta[(\logp^\theta_w-\logp^{ref}_w)-(\logp^\theta_l-\logp^{ref}_l)]\Big)
$$

无需在线采样、无需价值网络，但需要常驻一份冻结参考模型。

### 2.5 ORPO

ORPO **单模型、无参考模型**，同一次反向传播做 SFT + 偏好对齐。先把序列平均
对数概率 :math:`\bar\ell` 转成概率 :math:`p=\exp(\bar\ell)`，再取对数赔率：

$$
\log O(y|x)=\bar\ell-\log(1-\exp(\bar\ell))
$$

$$
\mathcal{L}_{OR}=-\frac{1}{B}\sum_i\log\sigma\big(\beta[\log O(y_i^w|x_i)-\log O(y_i^l|x_i)]\big)
$$

$$
\mathcal{L}_{ORPO}=\mathcal{L}_{SFT}+\lambda_{OR}\mathcal{L}_{OR}
$$

### 2.6 GRPO

对每个 prompt 采样 :math:`G` 条回答，用组内奖励相对高低当优势，**去掉价值网络**：

$$
A_i=\frac{r_i-\mathrm{mean}_{j}r_j}{\mathrm{std}_{j}r_j+\varepsilon}
$$

策略目标沿用 PPO 裁剪代理，无价值损失：

$$
\mathcal{L}_{GRPO}=-\frac{1}{G}\sum_i\frac{1}{|\mathcal{M}_i|}\sum_{t}
\min(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1\mp\varepsilon)A_i)
+\beta_{KL}\mathbb{D}_{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

### 2.7 四条路线数据流与 tradeoff 对比

| 维度 | PPO | DPO | ORPO | GRPO |
|---|---|---|---|---|
| 参考模型 | 需要（冻结） | 需要（冻结） | **不需要** | 可选（KL 惩罚用） |
| 价值网络 | 需要（value head） | 不需要 | 不需要 | **不需要** |
| 在线采样 | 需要（rollout 闭环） | 不需要（离线偏好对） | 不需要（离线偏好对） | 需要（组内采样 G 条） |
| 奖励来源 | 训练好的 RM 打分 | 偏好对隐式奖励 | 偏好对赔率比 | 规则 / 可验证奖励或 RM |
| 显存开销 | 最高（四模型） | 中（策略 + 参考） | 最低（单模型） | 中（策略 + 可选参考，无 critic） |
| 实现复杂度 | 高（GAE、裁剪、KL、奖励白化） | 低（成对 logits） | 低（SFT + OR） | 中（组采样、组内标准化） |
| 主要 tradeoff | 上限高但不稳定、调参多 | 稳定简单，但受离线偏好分布限制 | 省参考模型显存，赔率尺度需调 β/λ | 省 critic，依赖组内奖励区分度；奖励全同时优势退化 |

### 2.8 SimPO / KTO 简述

- **SimPO**（Meng et al., 2024）：DPO 思路上进一步去掉参考模型，用平均序列
  对数概率做隐式奖励，并引入长度归一化与目标 margin；本仓库未单独实现，留作
  扩展阅读。
- **KTO**（Ethayarajh et al., 2024）：不要求成对偏好，逐条接受 / 拒绝信号即可
  训练，损失按好 / 坏样本分别构造 logistic 项；适合只有隐式反馈（点赞 / 点踩）
  的场景。本仓库未单独实现。

## 3. 关键代码精读

### 3.1 序列对数概率提取 — `utils.gather_logprobs`

`src/llama31_8b_scratch/align/utils.py`：

```python
log_probs = F.log_softmax(logits, dim=-1)
mask = labels != ignore_index
labels_safe = labels.masked_fill(~mask, 0)
gathered = log_probs.gather(dim=-1, index=labels_safe.unsqueeze(-1)).squeeze(-1)
token_logps = gathered * mask
```

先 log-softmax 再按标签 gather；忽略位置填 0，避免非法 id 与污染序列求和。

### 3.2 成对损失 — `reward.pairwise_loss`

`src/llama31_8b_scratch/align/reward.py`：

```python
diff = chosen_scores - rejected_scores - margin
loss = -F.logsigmoid(diff).mean()
```

`logsigmoid` 内部对正负输入分流，大负差值也不会下溢到不可用范围。

### 3.3 GAE 递推 — `ppo.compute_gae`

`src/llama31_8b_scratch/align/ppo.py`，从后向前递推：

```python
delta = rewards[:, t] + gamma * next_value * non_terminal - values[:, t]
last_gae = delta + gamma * lam * non_terminal * last_gae
advantages[:, t] = last_gae
returns = advantages + values
```

终止步 `non_terminal=1-d_t` 切断向后引导；末尾 `V_T=0` 不做外部 bootstrap。

### 3.4 DPO 隐式奖励 — `dpo.dpo_loss`

`src/llama31_8b_scratch/align/dpo.py`：

```python
rewards_chosen = beta * (policy_chosen_logps - ref_chosen_logps)
rewards_rejected = beta * (policy_rejected_logps - ref_rejected_logps)
dpo_logits = rewards_chosen - rewards_rejected
loss = -F.logsigmoid(dpo_logits).mean()
```

### 3.5 ORPO 赔率比 — `orpo.orpo_loss`

`src/llama31_8b_scratch/align/orpo.py`：

```python
p = torch.exp(log_probs).clamp(max=1.0 - eps)
log_odds = log_probs - torch.log1p(-p)
odds_ratio_logits = beta * (log_odds_chosen - log_odds_rejected)
or_loss = -F.logsigmoid(odds_ratio_logits).mean()
total = chosen_nll + lambda_or * or_loss
```

### 3.6 GRPO 组内优势 — `grpo.compute_grpo_advantages`

`src/llama31_8b_scratch/align/grpo.py`：

```python
grouped = rewards.view(num_prompts, group_size)
group_mean = grouped.mean(dim=1, keepdim=True)
group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
advantages = (grouped - group_mean) / (group_std + eps)
```

### 3.7 工业封装 — `trl_wrappers`

`src/llama31_8b_scratch/align/trl_wrappers.py` 提供 `TRLPPOWrapper / TRLDPOWrapper /
TRLORPOWrapper / TRLGRPOWrapper`，`build_kwargs()` 做 YAML→TRL 字段映射，
`create_trainer()` 内部惰性 `import trl`；未装 trl 时模块仍可 import。

## 4. 运行命令

### 4.1 CPU 冒烟（单测同尺度，分钟级）

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch

python3 05_align/run_rm.py   --config configs/align/rm_smoke.yaml
python3 05_align/run_dpo.py   --config configs/align/dpo_smoke.yaml
python3 05_align/run_ppo.py   --config configs/align/ppo_smoke.yaml
python3 05_align/run_orpo.py  --config configs/align/orpo_smoke.yaml
python3 05_align/run_grpo.py  --config configs/align/grpo_smoke.yaml
```

命令行覆盖示例（可重复传入 `--override`）：

```bash
python3 05_align/run_dpo.py --config configs/align/dpo_smoke.yaml \
    --override align.beta=0.05 --override trainer.batch_size=4
```

### 4.2 8B 真机（多卡 GPU，bf16）

```bash
python3 05_align/run_rm.py   --config configs/align/rm_8b.yaml
python3 05_align/run_dpo.py   --config configs/align/dpo_8b.yaml
python3 05_align/run_ppo.py   --config configs/align/ppo_8b.yaml
python3 05_align/run_orpo.py  --config configs/align/orpo_8b.yaml
python3 05_align/run_grpo.py  --config configs/align/grpo_8b.yaml
```

### 4.3 单元测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "align_*.py" -v
```

> 说明：当前入口脚本只完成配置解析与 TRL 封装构造；backbone / 数据集对接
> model、data 代理交付后接入（脚本内标注 TBD）。CPU 闭式数值验证全部在
> `tests/core/align_*.py` 中完成；真机评测分数等无法在本环境实测的指标一律
> 记为 TBD，不编造数字。

## 5. 常见坑与排查

1. **`import trl` 报 ModuleNotFoundError**：本仓库把 trl/peft/accelerate 列为
   可选重依赖，只在 `create_trainer()` 内部惰性导入。核心链路（手写损失、单测）
   不依赖它们；真机训练前先 `pip install trl peft accelerate datasets`。
   若顶层就报 trl 缺失，说明重依赖被提到了模块顶层 import，需移进方法内部。

2. **DPO/ORPO 序列对数概率用“求和”还是“平均”**：DPO 用序列**求和** logp（参考
   模型抵消长度偏差）；ORPO 必须用**平均** token logp 再转赔率，否则长回答的
   对数概率绝对值更大，赔率被长度主导。两者不可混用，见 `orpo.sequence_mean_logprobs`。

3. **GRPO 组内奖励全相同导致除零**：某组 :math:`\mathrm{std}=0` 时优势应退化
   为全 0。`compute_grpo_advantages` 在分母加 `eps` 保护；若整批优势长期为 0，
   说明奖励函数对该批样本没有区分度，应检查奖励 / 采样多样性，而不是调裁剪系数。

4. **PPO 优势未白化导致策略损失尺度漂移**：不同 prompt 的奖励量纲差异大，
   直接喂优势会让大奖励样本主导梯度。`ppo_loss` 默认 `whiten_advantages=True`
   做批内标准化；KL 估计量在策略偏离较远时建议从 `naive`(k1) 切到 `k3`，
   避免朴素 KL 出现负值。

5. **标签 -100 进入对数概率求和**：prompt / padding 位置必须置 -100，
   `gather_logprobs` 才会掩码清零；若把 padding 当有效 token 求和，序列 logp
   会被大量高概率 padding token 抬高，偏好差被稀释。

## 下一期

- `06_eval`：对齐后模型的困惑度、多选 loglikelihood 与 lm-eval-harness 评测；
- `07_quant`：GPTQ / AWQ / 权重量化，把对齐后的模型压缩部署。

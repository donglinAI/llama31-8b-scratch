# 01 分词器：BPE 训练、Llama 3.1 对话模板与词表扩充

## 1. 本期解决什么问题

分词器是模型与文本之间的桥梁：把自然语言切分成模型词表中的 token id，推理时再
把 id 还原成文本。本期位于数据管道的最前端，输入是原始语料 jsonl，输出是可被
数据集与 collator 使用的分词器实例，以及下游统一依赖的 Llama 3.1 对话模板。

本期交付三类能力：

- **BPE 分词训练**：默认使用已安装的 `tokenizers`（Rust BPE）在自有语料上训练，
  SentencePiece / tiktoken 后端惰性预留；
- **对话模板**：`LlamaChatTemplate` 逐字实现 SPEC 5.4 模板，其他代理只调用、不
  自行拼字符串；
- **词表扩充**：不重训 BPE，为新增 token 分配 id 并同步扩展 embedding。

## 2. 核心原理

### 2.1 为什么不用整词词典

自然语言的开放词表问题：新词、人名、术语层出不穷，整词词典要么爆炸式膨胀，要么
大量词只能落进 `<unk>`。子词（subword）折中：常见词保留整词，稀有词拆成有意义的
片段，任意字符串都能被表示。

### 2.2 BPE 合并过程

Byte-Pair Encoding 从基础字母表（字节或字符）出发，反复统计相邻符号对的共现频次，
把最高频的一对合并成新 token。设第 $t$ 轮合并的符号对为 $(a_t, b_t)$，合并规则按
训练顺序固化进合并表 $M = \{m_1, m_2, \dots\}$。

训练目标（每轮贪心选择）：

$$
(a_t, b_t) = \arg\max_{(x,y)} \, \mathrm{count}(xy \mid \mathcal{D})
$$

其中 $\mathrm{count}(xy \mid \mathcal{D})$ 是语料 $\mathcal{D}$ 中相邻片段 $xy$
的出现次数。重复直到词表达到目标规模 $|V|$。

编码时，先把文本拆成基础符号，再按合并表顺序由低到高贪心合并；解码则按字节级
映射还原原始 UTF-8。字节级 BPE 的好处是不需要为未知字符准备多条回退路径，任意
字节都在基础字母表内。

### 2.3 Llama 3.1 模板结构

每一轮对话结构为：

```text
<|start_header_id|>{role}<|end_header_id|>

{content}<|eot_id|>
```

整个序列以 `<|begin_of_text|>` 开头。模板字符串如下（SPEC 5.4 逐字对齐）：

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>
```

官方 Llama 3.1 词表中特殊 token 的固定 id：

| 特殊 token | id | 含义 |
|---|---:|---|
| `<|begin_of_text|>` | 128000 | 序列起始 BOS |
| `<|end_of_text|>` | 128001 | 自然结束 EOS |
| `<|eot_id|>` | 128009 | 单轮结束 EOT |
| `<|finetune_right_pad_token|>` | 128004 | 右填充 PAD |
| `<|start_header_id|>` | 128006 | 角色头起始 |
| `<|end_header_id|>` | 128007 | 角色头结束 |

### 2.4 SFT 的 prompt 掩码

SFT 训练只在 assistant 回答上计算损失。设 prompt 段长度为 $L_p$，补全段长度为
$L_a$，对齐后的标签序列为：

$$
y_i =
\begin{cases}
-100, & i \le L_p \quad (\text{prompt 部分，损失忽略}) \\
\text{token id}_i, & i > L_p \quad (\text{assistant 部分})
\end{cases}
$$

标签与 `input_ids` 等长；序列移位由模型 loss 内部用
`F.cross_entropy(logits[..., :-1, :], labels[..., 1:])` 完成，数据集与 collator
不做移位。

### 2.5 词表扩充与 embedding 扩展

旧 embedding 形状 $W_{\text{old}} \in \mathbb{R}^{V_{\text{old}} \times D}$。
新增 $n$ 个 token 时，旧行原样保留，新行按截断正态初始化：

$$
w_{\text{new}} \sim \mathcal{N}(0, \sigma^2), \quad \sigma = \text{initializer\_range},
\quad \text{截断区间 } [-2\sigma, 2\sigma]
$$

扩展后 $W_{\text{new}} \in \mathbb{R}^{(V_{\text{old}}+n) \times D}$，旧 id 索引
不变，已有权重可继续加载。

## 3. 关键代码精读

### 3.1 对话模板：`src/llama31_8b_scratch/data/chat_template.py`

```python
def render_block(role: str, content: str) -> str:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"非法角色 {role!r}")
    return (
        f"{START_HEADER_TOKEN}{role}{END_HEADER_TOKEN}\n\n"
        f"{content}{EOT_TOKEN}"
    )
```

`LlamaChatTemplate.apply` 同时接受纯文本与消息列表；
`split_prompt_completion` 把 SFT 消息拆成 prompt 文本与 assistant 正文，供数据集
做 prompt 部分 -100 掩码。

### 3.2 BPE 训练：`src/llama31_8b_scratch/data/tokenizer_bpe.py`

```python
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    max_token_length=max_token_length,
    special_tokens=list(special_tokens),
)
tokenizer.train_from_iterator(list(texts), trainer=trainer)
tokenizer.add_special_tokens(list(special_tokens))
```

封装出的统一接口 `ScratchBPETokenizer` 提供 `encode / decode / encode_messages /
vocab_size / token_to_id / save / from_file`，三种后端实现同一契约。

### 3.3 数据集与掩码：`src/llama31_8b_scratch/data/datasets.py`

```python
prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
completion_ids = self.tokenizer.encode(completion_block, add_special_tokens=False)
input_ids = prompt_ids + completion_ids
labels = [-100] * len(prompt_ids) + list(completion_ids)
```

### 3.4 词表扩充：`src/llama31_8b_scratch/data/tokenizer_extend.py`

`extend_vocab` 负责去重合并词表并生成新 id 映射；`expand_embedding` 在 numpy 侧
拼接截断正态初始化的新行，`expand_embedding_torch` 提供 torch 版本。

### 3.5 入口脚本：`01_tokenizer/run_train_tokenizer.py`

薄入口只做配置加载与后端分发：默认 `tokenizers`，配置 `backend` 可切到
`sentencepiece` / `tiktoken`；重依赖在对应分支内部导入。

## 4. 运行命令

### 4.1 CPU 冒烟（toy 语料，默认 tokenizers 后端）

```bash
PYTHONPATH=src python3 01_tokenizer/run_train_tokenizer.py \
    --config configs/tokenizer/bpe_tokenizers.yaml
```

产物落在 `outputs/tokenizer/bpe/`：`tokenizer.json` 与 `special_tokens.json`。

### 4.2 覆盖配置项

```bash
PYTHONPATH=src python3 01_tokenizer/run_train_tokenizer.py \
    --config configs/tokenizer/bpe_tokenizers.yaml \
    --override train.vocab_size=500 \
    --override output_dir=outputs/tokenizer/toy
```

### 4.3 SentencePiece 后端（需先安装 sentencepiece）

```bash
pip install sentencepiece
PYTHONPATH=src python3 01_tokenizer/run_train_tokenizer.py \
    --config configs/tokenizer/sentencepiece.yaml
```

### 4.4 真机 8B 语料训练

```bash
# 替换为大规模纯文本/jsonl 语料，按词表规模调整 vocab_size
PYTHONPATH=src python3 01_tokenizer/run_train_tokenizer.py \
    --config configs/tokenizer/bpe_tokenizers.yaml \
    --override train.corpus_files='["data/raw/corpus.jsonl"]' \
    --override train.vocab_size=128256
```

### 4.5 运行数据相关单测

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "data_*.py" -v
```

## 5. 常见坑与排查

1. **特殊 token 被切成多个子词**
   - 现象：编码 `<|start_header_id|>` 后 token 列表里出现多个片段，角色头匹配失败。
   - 排查：训练后必须调用 `add_special_tokens` 二次登记，使特殊 token 整串识别；
     检查 `special_tokens` 列表是否与模板常量一致。

2. **中文解码出现乱码或重复字符**
   - 现象：decode 结果与原文不符。
   - 排查：确认预分词与解码器配对使用 ByteLevel；不要混用字符级预分词与字节级
     解码器。NFC 归一化保证同一字符只有一种编码表示。

3. **词表规模与预期不一致**
   - 现象：`vocab_size` 小于配置值。
   - 排查：小语料上低频子词对不足，BPE 提前停止合并；提高 `min_frequency` 观察
     变化，或扩充语料。特殊 token 会占用低位 id，计入总词表。

4. **`ImportError: sentencepiece` / `tiktoken`**
   - 现象：切换后端时报包缺失。
   - 排查：默认 tokenizers 路径不依赖这两个包；需要对应后端时按 00_setup 说明
     单独安装，未安装前不要在 CI 里强制跑 SP/tiktoken 分支。

5. **SFT 标签错位，回答部分被算进 prompt 损失**
   - 现象：模型只学会复述用户问题。
   - 排查：检查 `split_prompt_completion` 是否把最后一条 assistant 正确拆出；
     prompt 与 completion 分别编码后再拼接，labels 的 -100 长度必须与 prompt_ids
     长度严格相等。

## 下一期

- [02 模型结构：RMSNorm、RoPE、GQA 与 SwiGLU](../02_model/README.md)

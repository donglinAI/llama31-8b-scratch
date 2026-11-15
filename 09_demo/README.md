# 09 Demo：Gradio 网页对话与 OpenAI 客户端调用

## 1. 本期解决什么问题

服务起来了，但工程师才会用 curl。本期在全链路最后一环：给用户一个**能在浏览器里
对话**的网页，同时演示如何用标准库和官方 SDK 两种方式调用上一期的 API。

- **输入**：运行中的 OpenAI 兼容 API（08 期产出）；`configs/deploy/gradio.yaml`。
- **输出**：
  - 一个 Gradio 网页（多轮对话、流式逐字输出、temperature/top_p/max_tokens 滑块）；
  - 一个可直接运行的客户端示例脚本。
- **关键解耦**：网页**不碰模型权重**，只通过 HTTP 调 `/v1/chat/completions`。
  因此后端换成 vLLM/SGLang，网页一行不用改。

## 2. 核心原理

### 2.1 多轮对话 = 把历史拼成 messages

网页每发一轮，都要把"以前聊过的"重新发给模型（模型本身无状态）。Gradio 的历史是
`[(user, assistant), ...]`，我们在 `gradio_app.py::history_to_messages` 里把它
展开成：

```json
[
  {"role": "user", "content": "第一轮问题"},
  {"role": "assistant", "content": "第一轮回答"},
  {"role": "user", "content": "本轮问题"}
]
```

后端再用 Llama3.1 模板把这段 messages 渲染成 prompt（SPEC 5.4）。

### 2.2 流式渲染 = 读 SSE 流并增量拼接

网页要"打字机效果"，就必须开 `stream: true`。客户端逐行读 `data:` 帧，取
`delta.content` 累加：

$$\text{渲染文本}_t = \text{渲染文本}_{t-1} + \Delta_t$$

直到遇到 `data: [DONE]`。这与 08 期服务端的帧顺序一一对应，客户端不关心后端是
手写模型还是 vLLM。

### 2.3 为什么用 urllib 而不是直接装 openai 库

教学上先用**标准库 urllib** 把请求体、请求头、SSE 解析讲透（`client.py`），
再演示官方 SDK——后者只是把这些细节封装掉。两种写法打到同一个端点，行为一致。

## 3. 关键代码精读

| 文件 | 作用 |
|---|---|
| `09_demo/gradio_app.py` | `history_to_messages`（纯函数）+ 惰性 `build_demo` |
| `09_demo/openai_client_example.py` | 可运行的 urllib / SDK 示例 |
| `client.py` | `build_chat_request_body` / `stream_chat_messages` 复用实现 |

历史转 messages（`gradio_app.py`，纯函数，可单测）：

```python
def history_to_messages(history, user_msg):
    messages = []
    for turn in history or []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            if u: messages.append({"role": "user", "content": str(u)})
            if a: messages.append({"role": "assistant", "content": str(a)})
    messages.append({"role": "user", "content": user_msg})
    return messages
```

Gradio 回调是个生成器（逐 yield 累加文本即可实现逐字输出）：

```python
def respond(message, history, temperature, top_p, max_tokens):
    messages = history_to_messages(history, message)
    buf = ""
    for piece in stream_chat_messages(messages, model=model,
                                      base_url=api_url, ...):
        buf += piece
        yield buf
```

`gradio` 在 `build_demo` 内部才 import，模块顶层不依赖它。

## 4. 运行命令

### 先起 API（另一个终端）

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch
PYTHONPATH=src python3 08_deploy/run_api.py --config configs/deploy/api.yaml \
    --override model.device=cpu --override model.model_path=
```

### 起网页

```bash
pip install gradio>=4.26
PYTHONPATH=src python3 09_demo/gradio_app.py --config configs/deploy/gradio.yaml
# 浏览器打开 http://127.0.0.1:7860
```

### 跑客户端示例

```bash
PYTHONPATH=src python3 09_demo/openai_client_example.py
```

### Docker 一键（API + 网页）

```bash
docker compose up --build
# web: http://127.0.0.1:7860  api: http://127.0.0.1:8000
```

## 5. 常见坑与排查

1. **网页一直转圈但日志没报错**：多半是 API 没起，或 `gradio.yaml` 里 `api_url`
   指错了。先 `curl $api_url/v1/models` 确认连通。
2. **中文乱码**：urllib 读 SSE 行时用了 `ascii` 解码。本项目统一 `decode("utf-8")`，
   若你自己改了编码要改回来；请求体必须 `json.dumps(...).encode("utf-8")`。
3. **对话越聊越长、最后报错**：历史 messages 无截断，上下文超过 `max_model_len`。
   先在滑块上把 `max_tokens` 调小，后续可在前端加"只保留最近 N 轮"。
4. **Gradio 报 422 / 字段缺失**：新版 Gradio 的 `Chatbot` 历史格式从
   `(user, assistant)` 元组变成了 `{"role","content"}` 字典。`history_to_messages`
   两种都兼容，但你若自定义组件要确认喂进来的是哪种。
5. **`ConnectionRefused` 流式读一半断开**：默认 urllib 超时 120s，长生成请调大
   `urlopen(..., timeout=...)`，或改用 SDK 的流式上下文。

## 下一期

全链路到此贯通：数据 → 模型 → 训练 → 对齐 → 评测 → 量化 → 部署 → Demo。
回到 [README](../README.md) 看整体路线图，或继续打磨各期真机指标（真实分数 TBD）。

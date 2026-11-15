# 08 部署：OpenAI 兼容 API、vLLM/SGLang 与容器化

## 1. 本期解决什么问题

模型训练/对齐/量化完，最终要"被别人用起来"。本期在全链路里处于**量化之后、
网页 Demo 之前**：把一个模型包成一个网络服务，让任何客户端都能按统一协议调用。

- **输入**：一个 HF 格式（`model_type="llama"`）权重目录，或直接内存中的
  `ScratchForCausalLM`；以及 `configs/deploy/*.yaml`。
- **输出**：
  - 一个 OpenAI 兼容的 HTTP 服务：`POST /v1/chat/completions`、`GET /v1/models`，
    支持非流式与 SSE 流式；
  - vLLM / SGLang 两种工业引擎的启动封装；
  - Docker / docker-compose 一键起 API 与网页。
- **边界**：本期不做评测分数（那是 06），不做多机路由，只把"模型 → 服务"这条
  链路打通并保证协议逐字兼容 OpenAI。

## 2. 核心原理

### 2.1 为什么是 OpenAI 兼容协议

客户端生态（官方 SDK、LangChain、各类 UI）都只认 `POST /v1/chat/completions`。
只要我们在**路径、字段名、SSE 帧格式、`[DONE]` 结束标记**上逐字一致，客户端改个
`base_url` 就能用，不必为我们的模型单独写对接。

非流式响应骨架：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama31-8b-scratch",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

### 2.2 SSE 流式为什么这样切

流式时响应体是一个 `text/event-stream`，每段增量是一帧：

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"}}]}\n\n
...
data: [DONE]\n\n
```

帧顺序严格对齐 OpenAI：①首帧只给 `delta.role="assistant"`；②每个 token 一段
`delta.content`；③结束帧 `delta={}` 且 `finish_reason="stop"`；④可选的 usage 统计帧。
规范要求每行以 `data: `（冒号后一个空格）开头、帧间两个换行，本项目在
`deployment/sse.py` 里把这些格式固化成纯函数，单测逐字符断言。

### 2.3 手写后端 vs vLLM/SGLang

逐 token 解码是**访存受限**的：每生成一个 token 都要把全部权重从显存读一遍。

- 手写 `ScratchForCausalLM`（`deployment/service.py::ScratchCausalLMService`）
  教学清晰，但 batch 小、GPU 利用率低；
- **vLLM** 用 **PagedAttention** 把 KV Cache 切成定长 block（像操作系统虚拟内存
  分页），显存利用率从 <40% 提到 90%+，再加**连续批处理**让 GPU 不空等；
- **SGLang** 用 **RadixAttention**（基数树）自动复用相同前缀的 KV，对多轮对话、
  共享 system prompt / few-shot 尤其省。

三者实现同一个 `BaseModelService.generate / generate_stream`，API 层因此可无缝
切换——这就是统一抽象层的价值。

## 3. 关键代码精读

实质实现都在 `src/llama31_8b_scratch/deployment/`：

| 文件 | 作用 |
|---|---|
| `schemas.py` | pydantic 请求/响应/chunk 模型，含默认值与字段校验 |
| `sse.py` | `sse_frame` / `to_sse_json` / `sse_done` 纯函数 |
| `service.py` | `BaseModelService` 抽象 + `map_request_to_params` 纯映射 |
| `api_server.py` | `create_app(service)`；处理逻辑抽成纯函数，路由只是薄包装 |
| `vllm_serve.py` / `sglang_serve.py` | 纯函数拼启动参数 + 惰性 import 启动 |
| `client.py` | urllib 原生调用 + openai SDK 示例 |

请求→参数映射（`service.py`，纯、可单测）：

```python
def map_request_to_params(request, default_max_tokens=128):
    max_tokens = request.effective_max_tokens or default_max_tokens
    return GenerateParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=int(max_tokens),
        repetition_penalty=1.0 + request.frequency_penalty,  # 频率惩罚近似重复惩罚
        stop=_normalize(request.stop),
    )
```

流式 chunk 生成（`api_server.py::iter_chat_chunks`）：

```python
yield _chunk(model, rid, created, {"role": "assistant"})     # 首帧
for piece in service.generate_stream(prompt, params):
    yield _chunk(model, rid, created, {"content": piece})     # 内容帧
yield _chunk(model, rid, created, {}, finish_reason="stop")  # 结束帧
```

惰性导入是硬约束（`vllm_serve.py`）：

```python
def run_server(cfg):
    try:
        from vllm.entrypoints.openai.api_server import run_uvicorn
    except ImportError as exc:
        raise ImportError("请先 pip install vllm>=0.5") from exc
```

这样只装核心依赖时 `import llama31_8b_scratch.deployment` 也不会报错。

## 4. 运行命令

### CPU smoke（手写后端，随机权重，秒级）

```bash
cd /home/user/Doubao/chats/38453210918078466/llama31-8b-scratch
PYTHONPATH=src python3 08_deploy/run_api.py --config configs/deploy/api.yaml \
    --override model.device=cpu --override model.model_path=
```

另开一个终端验证：

```bash
curl http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"llama31-8b-scratch","messages":[{"role":"user","content":"你好"}]}'
```

### 8B 真机（手写后端 / vLLM）

```bash
# 手写后端直驱
PYTHONPATH=src python3 08_deploy/run_api.py --config configs/deploy/api.yaml \
    --override model.model_path=outputs/llama3_1_8b_pretrained

# vLLM 工业后端（推荐）
PYTHONPATH=src python3 -c "
from llama31_8b_scratch.common.config import load_config
from llama31_8b_scratch.deployment.vllm_serve import run_server
run_server(load_config('configs/deploy/vllm.yaml'))
"
```

### Docker

```bash
docker compose up --build api        # 起 API（:8000）
docker compose up --build web       # 起 Gradio 网页（:7860）
```

### 单测

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/core -p "deploy_*.py" -v
```

## 5. 常见坑与排查

1. **流式一直不返回、浏览器挂住**：响应头漏了 `Cache-Control: no-cache` 或中间有
   缓冲代理。本项目在 `StreamingResponse` 里同时设了 `X-Accel-Buffering: no`，
   经 Nginx 时也要关缓冲。
2. **客户端报 `422 Unprocessable Entity`**：多半是 `messages` 为空、`temperature`
   超出 0~2，或传了不被接受的字段。看 FastAPI 返回的 detail 列表逐项核对；多余字段
   已被 `extra="ignore"` 丢弃，只有必填/越界才报错。
3. **`data: [DONE]` 之后客户端还在等**：帧末尾必须是 `\\n\\n` 两个换行，且
   `[DONE]` 不能包成 JSON。用 `grep` 确认实际字节；少一个换行就会让解析器一直阻塞。
4. **vLLM 报 `model_type` 不认识**：我们的 `save_pretrained` 写出的 `config.json`
   必须是 `model_type="llama"`（SPEC 5.2）。检查权重目录里有没有 `config.json`，
   别只拷了 `model.safetensors`。
5. **显存 OOM**：`gpu_memory_utilization` 默认 0.9，多进程同卡时调低；`max_model_len`
   决定 KV Cache 上限，先按 4096 起再逐步放大。

## 下一期

[09 Demo](../09_demo/README.md)：给这个服务套一个能聊天的网页，并用 OpenAI 客户端调通。

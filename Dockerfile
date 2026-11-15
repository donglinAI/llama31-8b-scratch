# ============================================================
# llama31-8b-scratch 推理服务镜像
# 基础镜像：官方 nvidia/cuda 运行时（含 CUDA runtime），再装 Python
# 分层缓存：先拷依赖清单装依赖，再拷源码，改代码不重装依赖
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 避免交互前端卡住
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 安装 Python 与基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

# ---- 依赖层（改动最少，最大化 Docker 层缓存）----
COPY requirements.txt ./
# 部署/运行最小集：核心 + 服务端依赖；vllm/sglang/gradio 可在运行镜像里按需追加
RUN pip3 install --upgrade pip \
    && pip3 install fastapi "uvicorn[standard]" pydantic PyYAML torch --index-url https://download.pytorch.org/whl/cu124 || pip3 install fastapi "uvicorn[standard]" pydantic PyYAML

# ---- 源码层 ----
COPY src ./src
COPY configs ./configs
COPY 08_deploy ./08_deploy
COPY 09_demo ./09_demo

ENV PYTHONPATH=/workspace/src

# API 端口 / Gradio 端口
EXPOSE 8000 7860

# 默认启动 OpenAI 兼容 API（compose 里 web 会覆盖命令）
CMD ["python", "08_deploy/run_api.py", "--config", "configs/deploy/api.yaml", "--override", "model.model_path="]

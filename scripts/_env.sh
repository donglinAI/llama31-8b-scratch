#!/usr/bin/env bash
# ============================================================
# 统一环境脚本：所有启动脚本开头 `source scripts/_env.sh`
# 集中管理项目根目录、Python 路径、设备、端口等变量
# 本文件只做变量定义与路径推导，不执行任何训练/推理动作
# ============================================================

# 本文件位于 <root>/scripts/_env.sh，由此反推项目根目录
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRATCH_ROOT="$(cd "${_ENV_SH_DIR}/.." && pwd)"
export SCRATCH_PYTHON="${SCRATCH_PYTHON:-python3}"
export PYTHONPATH="${SCRATCH_ROOT}/src:${PYTHONPATH:-}"

# 运行设备：auto / cpu / cuda
export SCRATCH_DEVICE="${SCRATCH_DEVICE:-auto}"

# 服务默认端口
export SCRATCH_API_PORT="${SCRATCH_API_PORT:-8000}"
export SCRATCH_WEB_PORT="${SCRATCH_WEB_PORT:-7860}"

# 输出与数据目录
export SCRATCH_OUTPUTS_DIR="${SCRATCH_ROOT}/outputs"
export SCRATCH_DATA_DIR="${SCRATCH_ROOT}/data"
export SCRATCH_TOY_DIR="${SCRATCH_ROOT}/data/toy"

mkdir -p "${SCRATCH_OUTPUTS_DIR}"

# 可选：加载本地 .env（不入库）
if [ -f "${SCRATCH_ROOT}/.env" ]; then
    # shellcheck disable=SC1091
    set -a
    source "${SCRATCH_ROOT}/.env"
    set +a
fi

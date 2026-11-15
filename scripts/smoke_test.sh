#!/usr/bin/env bash
# ============================================================
# 全链路冒烟测试：CPU 小样本，依次走通
# 分词 -> 模型自检 -> 预训练 -> 指令微调 -> DPO 对齐
# 任一阶段失败立即退出并返回非零码，便于在 CI / 视频里给出可验证证据
# ============================================================
set -euo pipefail

# 统一环境：SCRATCH_ROOT / PYTHONPATH / 端口等
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
cd "${SCRATCH_ROOT}"

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

banner "1/5 训练 BPE 分词器（词表 512）"
"${SCRATCH_PYTHON}" 01_tokenizer/run_train_tokenizer.py \
    --config configs/tokenizer/smoke.yaml

banner "2/5 模型结构自检（前向 / 落盘 / 回填 / 生成）"
"${SCRATCH_PYTHON}" 02_model/run_model_check.py \
    --config configs/model/smoke.yaml

banner "3/5 预训练（50 个优化步，CPU）"
"${SCRATCH_PYTHON}" 03_pretrain/run_pretrain.py \
    --config configs/pretrain/smoke.yaml

banner "4/5 指令微调 SFT（30 个优化步，CPU）"
"${SCRATCH_PYTHON}" 04_sft/run_sft.py \
    --config configs/sft/smoke.yaml

banner "5/5 DPO 对齐（10 个优化步，CPU）"
"${SCRATCH_PYTHON}" 05_align/run_dpo.py \
    --config configs/align/dpo_smoke.yaml

echo ""
echo "✅ 全链路冒烟通过：分词 → 模型 → 预训练 → SFT → DPO"
echo "   产物位于 outputs/，可继续运行 06~09 的评测 / 量化 / 部署 / Demo"

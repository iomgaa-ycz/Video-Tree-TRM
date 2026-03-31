#!/bin/bash
# Phase 2: ACT 训练（多轮 max_rounds=5，NavigationLoss + ACTLoss）
# 用法: bash scripts/train_phase2.sh [EPOCHS]
#
# 前置条件: Phase 1 已完成，存在 data/videomme/checkpoints/phase1_best.pt
#
# 示例:
#   bash scripts/train_phase2.sh        # 默认 20 epochs
#   bash scripts/train_phase2.sh 30     # 30 epochs

set -euo pipefail

EPOCHS=${1:-20}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CKPT_PATH="data/videomme/checkpoints/phase1_best.pt"

# 检查 Phase 1 检查点
if [ ! -f "${CKPT_PATH}" ]; then
    echo "错误: Phase 1 检查点不存在: ${CKPT_PATH}"
    echo "请先完成 Phase 1 训练"
    exit 1
fi

echo "============================================"
echo " Phase 2: ACT 训练"
echo "============================================"
echo " Phase 1 检查点: ${CKPT_PATH}"
echo " Phase 2 Epochs: ${EPOCHS}"
echo " max_rounds:     5"
echo " 时间戳:         ${TIMESTAMP}"
echo "============================================"

nohup conda run -n Video-Tree-TRM python train.py \
    --config config/videomme.yaml \
    --set train.max_epochs_phase1=0 \
    --set train.max_epochs_phase2="${EPOCHS}" \
    > "logs/train_phase2_${TIMESTAMP}.log" 2>&1 &

PID=$!
echo ""
echo "训练进程已启动, PID: ${PID}"
echo "日志文件: logs/train_phase2_${TIMESTAMP}.log"
echo ""
echo "监控命令:"
echo "  tail -f logs/train_phase2_${TIMESTAMP}.log"
echo "  grep 'Phase 2' logs/system.log | tail -20"
echo "  ps -p ${PID}"

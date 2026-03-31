#!/bin/bash
# Phase 1: 导航训练（单轮 max_rounds=1，仅 NavigationLoss）
# 用法: bash scripts/train_phase1.sh [EPOCHS]
#
# 示例:
#   bash scripts/train_phase1.sh        # 默认 30 epochs
#   bash scripts/train_phase1.sh 50     # 50 epochs

set -euo pipefail

EPOCHS=${1:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " Phase 1: 导航训练 (max_rounds=1)"
echo "============================================"
echo " Epochs: ${EPOCHS}"
echo " 时间戳: ${TIMESTAMP}"
echo "============================================"

nohup conda run -n Video-Tree-TRM python train.py \
    --config config/videomme.yaml \
    --set train.max_epochs_phase1="${EPOCHS}" \
    --set train.max_epochs_phase2=0 \
    > "logs/train_phase1_${TIMESTAMP}.log" 2>&1 &

PID=$!
echo ""
echo "训练进程已启动, PID: ${PID}"
echo "日志文件: logs/train_phase1_${TIMESTAMP}.log"
echo ""
echo "监控命令:"
echo "  tail -f logs/train_phase1_${TIMESTAMP}.log"
echo "  grep 'Phase 1' logs/system.log | tail -20"
echo "  ps -p ${PID}"

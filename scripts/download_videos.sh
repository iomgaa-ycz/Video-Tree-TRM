#!/usr/bin/env bash
# =============================================================================
# download_videos.sh — 批量下载 VideoMME 长视频（MP4 含音轨）
#
# 用法:
#   bash scripts/download_videos.sh        # 下载全部未完成的视频
#
# 特性:
#   - 已存在的 mp4 自动跳过（断点续传）
#   - bestvideo+bestaudio 合并，保证有音轨
#   - 并发数: 3（避免被 YouTube 限流）
#   - 失败记录写入 logs/failed_downloads.txt
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JSONL="${PROJECT_ROOT}/data/videomme/metadata/long_videos.jsonl"
VIDEO_DIR="${PROJECT_ROOT}/data/videomme/videos"
LOG_DIR="${PROJECT_ROOT}/data/videomme/logs"
LOG="${LOG_DIR}/download_$(date +%Y%m%d_%H%M%S).log"
FAILED_FILE="${LOG_DIR}/failed_downloads.txt"
CONCURRENT=3

mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"

# 激活 conda 环境（yt-dlp、ffmpeg 在此环境中）
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Video-Tree-TRM

echo "[$(date)] ===== 批量下载开始 =====" | tee "${LOG}"
echo "[$(date)] 视频目录: ${VIDEO_DIR}" | tee -a "${LOG}"

# 读取所有 youtube_id，写入临时文件供 xargs 读取
TMP_IDS=$(mktemp)
python3 -c "
import json
with open('${JSONL}') as f:
    for line in f:
        d = json.loads(line.strip())
        print(d['youtube_id'])
" > "${TMP_IDS}"

TOTAL=$(wc -l < "${TMP_IDS}")
echo "[$(date)] 总计: ${TOTAL} 个视频，并发数: ${CONCURRENT}" | tee -a "${LOG}"

# 并发下载：每个 youtube_id 单独调用 yt-dlp
# 注意：直接在 xargs 中内联参数，不依赖 bash 数组导出
xargs -P "${CONCURRENT}" -I YID bash -c '
    VIDEO_DIR="'"${VIDEO_DIR}"'"
    LOG="'"${LOG}"'"
    FAILED_FILE="'"${FAILED_FILE}"'"
    YID="YID"
    OUT="${VIDEO_DIR}/${YID}.mp4"

    if [[ -f "${OUT}" && -s "${OUT}" ]]; then
        echo "[$(date)] [SKIP] ${YID}" >> "${LOG}"
        exit 0
    fi

    URL="https://www.youtube.com/watch?v=${YID}"
    echo "[$(date)] [START] ${YID}" >> "${LOG}"

    if trickle -d 785 yt-dlp \
        --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best" \
        --merge-output-format mp4 \
        --output "${OUT}" \
        --no-playlist \
        --retries 5 \
        --fragment-retries 5 \
        --socket-timeout 60 \
        --no-warnings \
        "${URL}" >> "${LOG}" 2>&1; then
        echo "[$(date)] [OK]   ${YID}  size=$(du -sh "${OUT}" 2>/dev/null | cut -f1)" >> "${LOG}"
    else
        echo "[$(date)] [FAIL] ${YID}" >> "${LOG}"
        echo "${YID}" >> "${FAILED_FILE}"
    fi
' < "${TMP_IDS}"

rm -f "${TMP_IDS}"

TOTAL_MP4=$(find "${VIDEO_DIR}" -name "*.mp4" -size +0c | wc -l)
FAILED_COUNT=$(wc -l < "${FAILED_FILE}" 2>/dev/null || echo 0)

echo "" | tee -a "${LOG}"
echo "[$(date)] ===== 汇总 =====" | tee -a "${LOG}"
echo "[$(date)] 目录共 MP4: ${TOTAL_MP4} / ${TOTAL}" | tee -a "${LOG}"
[[ "${FAILED_COUNT}" -gt 0 ]] && echo "[$(date)] 失败数: ${FAILED_COUNT}，列表: ${FAILED_FILE}" | tee -a "${LOG}"
echo "[$(date)] 日志: ${LOG}" | tee -a "${LOG}"

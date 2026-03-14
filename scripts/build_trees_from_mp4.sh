#!/usr/bin/env bash
# =============================================================================
# build_trees_from_mp4.sh — 从本地 MP4 批量建树（JSON 输出，无 embedding）
#
# 用法:
#   bash scripts/build_trees_from_mp4.sh
#   bash scripts/build_trees_from_mp4.sh data/videomme/videos/TGom0uiW130.mp4  # 单文件
#
# 环境变量:
#   VIDEO_DIR  — MP4 目录（默认: <PROJECT_ROOT>/data/videomme/videos）
#   CONFIG     — 配置文件（默认: config/videomme.yaml）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/videomme/videos}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/videomme.yaml}"
TREE_DIR="${PROJECT_ROOT}/data/videomme/trees"
LOG_DIR="${PROJECT_ROOT}/data/videomme/logs"
LOG="${LOG_DIR}/mp4_build_$(date +%Y%m%d_%H%M%S).log"
FAILED="${LOG_DIR}/failed_mp4_builds.txt"

mkdir -p "${TREE_DIR}" "${LOG_DIR}"

# 绕过代理
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}100.83.164.94"
export no_proxy="${no_proxy:+${no_proxy},}100.83.164.94"

# 激活 conda 环境
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Video-Tree-TRM
cd "${PROJECT_ROOT}"

# 确定待处理文件列表
if [[ $# -gt 0 ]]; then
    MP4_FILES=("$@")
else
    mapfile -t MP4_FILES < <(find "${VIDEO_DIR}" -name "*.mp4" | sort)
fi

TOTAL=${#MP4_FILES[@]}
BUILT=0; SKIP=0; FAIL=0

echo "[$(date)] ===== MP4 本地建树开始 =====" | tee "${LOG}"
echo "[$(date)] 待处理: ${TOTAL} 个视频" | tee -a "${LOG}"
echo "[$(date)] 配置: ${CONFIG}" | tee -a "${LOG}"

for MP4 in "${MP4_FILES[@]}"; do
    [[ -z "${MP4}" ]] && continue
    STEM="$(basename "${MP4}" .mp4)"
    CACHE="${TREE_DIR}/${STEM}_video.json"

    # 缓存命中跳过
    if [[ -f "${CACHE}" ]]; then
        SKIP=$((SKIP+1))
        echo "[$(date)] [SKIP] ${STEM}" | tee -a "${LOG}"
        continue
    fi

    echo "[$(date)] [BUILD] ${STEM}  file=${MP4}" | tee -a "${LOG}"

    if python -u main.py index \
        --source "${MP4}" \
        --modality video \
        --config "${CONFIG}" \
        >> "${LOG}" 2>&1; then
        BUILT=$((BUILT+1))
        echo "[$(date)] [OK]   ${STEM}" | tee -a "${LOG}"
    else
        FAIL=$((FAIL+1))
        echo "[$(date)] [FAIL] ${STEM}" | tee -a "${LOG}"
        echo "${STEM} ${MP4}" >> "${FAILED}"
    fi
done

TREE_COUNT=$(find "${TREE_DIR}" -name "*_video.json" 2>/dev/null | wc -l)
echo "" | tee -a "${LOG}"
echo "[$(date)] ===== 汇总 =====" | tee -a "${LOG}"
echo "[$(date)] 本次新建: ${BUILT}  跳过: ${SKIP}  失败: ${FAIL}" | tee -a "${LOG}"
echo "[$(date)] 树索引总数: ${TREE_COUNT}" | tee -a "${LOG}"
[[ ${FAIL} -gt 0 ]] && echo "[$(date)] 失败列表: ${FAILED}" | tee -a "${LOG}"
echo "[$(date)] 日志: ${LOG}" | tee -a "${LOG}"

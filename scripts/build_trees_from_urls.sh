#!/usr/bin/env bash
# =============================================================================
# build_trees_from_urls.sh — 直接从 YouTube URL 批量建树（不下载视频）
#
# 用法:
#   # 全量（300 个长视频）
#   bash scripts/build_trees_from_urls.sh
#
#   # 只处理前 N 个
#   head -10 data/videomme/metadata/long_videos.jsonl \
#       | bash scripts/build_trees_from_urls.sh --stdin
#
# 环境变量:
#   DATA_DIR   — 数据根目录（默认: <PROJECT_ROOT>/data/videomme）
#   CONFIG     — 配置文件路径（默认: config/videomme.yaml）
#
# 特性:
#   - 自动激活 Video-Tree-TRM conda 环境
#   - 缓存命中跳过（trees/{youtube_id}_video.pkl 已存在则跳过）
#   - 断点续传（重复运行安全）
#   - 失败记录写入 logs/failed_url_builds.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# 0. 全局配置
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/videomme}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/videomme.yaml}"
JSONL="${DATA_DIR}/metadata/long_videos.jsonl"
TREE_DIR="${DATA_DIR}/trees"
LOG_DIR="${DATA_DIR}/logs"
LOG="${LOG_DIR}/url_build_$(date +%Y%m%d_%H%M%S).log"
FAILED="${LOG_DIR}/failed_url_builds.txt"

mkdir -p "${TREE_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# 0.5 绕过代理：GPU 内网地址直连，不经过 http_proxy
# ---------------------------------------------------------------------------
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}100.83.164.94"
export no_proxy="${no_proxy:+${no_proxy},}100.83.164.94"

# ---------------------------------------------------------------------------
# 1. 激活 conda 环境
# ---------------------------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Video-Tree-TRM
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 2. 解析参数
# ---------------------------------------------------------------------------
STDIN_MODE=false
for arg in "$@"; do
    [[ "${arg}" == "--stdin" ]] && STDIN_MODE=true
done

# ---------------------------------------------------------------------------
# 3. 读取输入
# ---------------------------------------------------------------------------
if [[ "${STDIN_MODE}" == true ]]; then
    # 从 stdin 读取（支持 head -N | bash ... --stdin）
    INPUT_DATA="$(cat)"
else
    INPUT_DATA="$(cat "${JSONL}")"
fi

TOTAL=$(echo "${INPUT_DATA}" | wc -l)
BUILT=0; SKIP=0; FAIL=0

echo "[$(date)] ===== URL 流式建树开始 =====" | tee "${LOG}"
echo "[$(date)] 待处理: ${TOTAL} 个长视频" | tee -a "${LOG}"
echo "[$(date)] 配置: ${CONFIG}" | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# 4. 主循环
# ---------------------------------------------------------------------------
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue

    YID=$(python -c "import sys,json; print(json.loads(sys.argv[1])['youtube_id'])" "${line}")
    URL=$(python -c "import sys,json; print(json.loads(sys.argv[1])['url'])"        "${line}")
    CACHE="${TREE_DIR}/${YID}_video.json"

    # 缓存命中跳过
    if [[ -f "${CACHE}" ]]; then
        SKIP=$((SKIP+1))
        echo "[$(date)] [SKIP] ${YID}" | tee -a "${LOG}"
        continue
    fi

    echo "[$(date)] [BUILD] ${YID}  url=${URL}" | tee -a "${LOG}"

    if python main.py index \
        --source "${URL}" \
        --modality video \
        --config "${CONFIG}" \
        >> "${LOG}" 2>&1; then
        BUILT=$((BUILT+1))
        echo "[$(date)] [OK]   ${YID}" | tee -a "${LOG}"
    else
        FAIL=$((FAIL+1))
        echo "[$(date)] [FAIL] ${YID}" | tee -a "${LOG}"
        echo "${YID} ${URL}" >> "${FAILED}"
    fi

done <<< "${INPUT_DATA}"

# ---------------------------------------------------------------------------
# 5. 汇总
# ---------------------------------------------------------------------------
TREE_COUNT=$(find "${TREE_DIR}" -name "*_video.json" 2>/dev/null | wc -l)
echo "" | tee -a "${LOG}"
echo "[$(date)] ===== 汇总 =====" | tee -a "${LOG}"
echo "[$(date)] 本次新建: ${BUILT}  跳过: ${SKIP}  失败: ${FAIL}" | tee -a "${LOG}"
echo "[$(date)] 树索引总数: ${TREE_COUNT} / ${TOTAL}" | tee -a "${LOG}"
[[ ${FAIL} -gt 0 ]] && echo "[$(date)] 失败列表: ${FAILED}" | tee -a "${LOG}"
echo "[$(date)] 日志: ${LOG}" | tee -a "${LOG}"

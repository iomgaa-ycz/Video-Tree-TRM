#!/usr/bin/env bash
# =============================================================================
# build_videomme_trees.sh — VideoMME 长视频数据预处理：下载 + 建树
# =============================================================================
# 功能:
#   1. 初始化目录结构 (/data/videomme/...)
#   2. 激活 Conda 环境 (Video-Tree-TRM)
#   3. 安装必要工具 (yt-dlp, datasets)
#   4. 从 HuggingFace 下载 VideoMME 元数据，提取长视频列表
#   5. 用 yt-dlp 批量下载长视频（断点续传，跳过已下载）
#   6. 为每个视频调用 main.py index 建树（跳过已缓存）
#   7. 汇总日志
#
# 使用方式:
#   cd /home/undergraduate/Video-Tree-TRM
#   bash scripts/build_videomme_trees.sh
#
#   可选环境变量覆盖:
#   DATA_DIR=/other/path bash scripts/build_videomme_trees.sh
#   WORKERS=4 bash scripts/build_videomme_trees.sh   # 并行建树进程数
#
# 断点续传:
#   重复运行完全安全 —— yt-dlp 跳过已下载文件，main.py 跳过缓存命中的树。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. 全局配置
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-Video-Tree-TRM}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/videomme}"
VIDEO_DIR="${DATA_DIR}/videos"
META_DIR="${DATA_DIR}/metadata"
TREE_DIR="${DATA_DIR}/trees"
LOG_DIR="${DATA_DIR}/logs"
CKPT_DIR="${DATA_DIR}/checkpoints"

CONFIG_YAML="${PROJECT_ROOT}/config/videomme.yaml"
ENV_FILE="${PROJECT_ROOT}/.env"
META_SCRIPT="${SCRIPT_DIR}/_download_meta.py"

WORKERS="${WORKERS:-1}"                  # 并行建树进程数（默认串行，保护 API 速率）
MIN_DURATION="${MIN_DURATION:-1800}"     # 长视频最短时长（秒）
MAX_DURATION="${MAX_DURATION:-3600}"     # 长视频最长时长（秒）
YTDLP_RATE="${YTDLP_RATE:-500K}"        # yt-dlp 下载限速（防封）
YTDLP_RETRIES="${YTDLP_RETRIES:-5}"     # yt-dlp 重试次数

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/build_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*" | tee -a "${MAIN_LOG}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*" | tee -a "${MAIN_LOG}"; }
error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" | tee -a "${MAIN_LOG}"; }

# ---------------------------------------------------------------------------
# 1. 创建目录结构
# ---------------------------------------------------------------------------

info "=== Step 1: 初始化目录结构 ==="
mkdir -p "${VIDEO_DIR}" "${META_DIR}" "${TREE_DIR}" "${LOG_DIR}" "${CKPT_DIR}"
info "数据目录已就绪: ${DATA_DIR}"
info "  videos/      → ${VIDEO_DIR}"
info "  metadata/    → ${META_DIR}"
info "  trees/       → ${TREE_DIR}"
info "  logs/        → ${LOG_DIR}"
info "  checkpoints/ → ${CKPT_DIR}"

# ---------------------------------------------------------------------------
# 2. 激活 Conda 环境
# ---------------------------------------------------------------------------

info "=== Step 2: 激活 Conda 环境 (${CONDA_ENV}) ==="

# 找到 conda 初始化脚本
CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
if [[ -z "${CONDA_BASE}" ]]; then
    error "未找到 conda，请确保 conda 已安装并在 PATH 中"
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
info "已激活环境: $(conda info --envs | grep '*' | awk '{print $1}')"
info "Python 路径: $(which python)"

# ---------------------------------------------------------------------------
# 3. 安装必要工具
# ---------------------------------------------------------------------------

info "=== Step 3: 安装必要工具 ==="

pip install --quiet --upgrade yt-dlp datasets
info "yt-dlp 版本: $(yt-dlp --version)"
python -c "import datasets; print(f'datasets 版本: {datasets.__version__}')"

# ---------------------------------------------------------------------------
# 4. 下载 VideoMME 元数据，提取长视频列表
# ---------------------------------------------------------------------------

info "=== Step 4: 下载 VideoMME 元数据 ==="

LONG_VIDEOS_JSONL="${META_DIR}/long_videos.jsonl"
LONG_QA_JSONL="${META_DIR}/long_videos_qa.jsonl"

if [[ -f "${LONG_VIDEOS_JSONL}" ]]; then
    EXISTING_COUNT=$(wc -l < "${LONG_VIDEOS_JSONL}")
    warn "元数据已存在 (${EXISTING_COUNT} 条长视频)，跳过下载。如需重新下载，删除 ${LONG_VIDEOS_JSONL}"
else
    # 配置 HuggingFace 镜像（国内加速，如已能直连可注释掉）
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    info "HuggingFace 端点: ${HF_ENDPOINT}"

    python "${META_SCRIPT}" \
        --meta-dir "${META_DIR}" \
        --min-duration "${MIN_DURATION}" \
        --max-duration "${MAX_DURATION}" \
        2>&1 | tee -a "${MAIN_LOG}"
fi

# 确认文件存在
if [[ ! -f "${LONG_VIDEOS_JSONL}" ]]; then
    error "元数据文件不存在，元数据下载失败: ${LONG_VIDEOS_JSONL}"
    exit 1
fi

TOTAL_VIDEOS=$(wc -l < "${LONG_VIDEOS_JSONL}")
info "长视频总数: ${TOTAL_VIDEOS}"

# ---------------------------------------------------------------------------
# 5. 批量下载长视频（yt-dlp，断点续传）
# ---------------------------------------------------------------------------

info "=== Step 5: 批量下载长视频 ==="
info "下载目录: ${VIDEO_DIR}"
info "限速: ${YTDLP_RATE}，重试次数: ${YTDLP_RETRIES}"

DOWNLOAD_LOG="${LOG_DIR}/download_${TIMESTAMP}.log"
FAILED_DOWNLOADS="${LOG_DIR}/failed_downloads_${TIMESTAMP}.txt"
DOWNLOAD_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS= read -r line; do
    VIDEO_ID=$(echo "${line}" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('youtube_id') or d.get('video_id',''))")
    URL=$(echo "${line}" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('url',''))")

    if [[ -z "${VIDEO_ID}" || -z "${URL}" ]]; then
        warn "跳过无效记录: ${line:0:80}"
        continue
    fi

    # 检查是否已下载（任意格式均算）
    EXISTING_FILE=$(find "${VIDEO_DIR}" -name "${VIDEO_ID}.*" -type f 2>/dev/null | head -1)
    if [[ -n "${EXISTING_FILE}" ]]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    info "[下载 ${DOWNLOAD_COUNT}/${TOTAL_VIDEOS}] ${VIDEO_ID}"

    # yt-dlp 下载：优先 mp4，最高 720p（节省空间），单文件断点续传
    if yt-dlp \
        --output "${VIDEO_DIR}/%(id)s.%(ext)s" \
        --format "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best" \
        --merge-output-format mp4 \
        --retries "${YTDLP_RETRIES}" \
        --rate-limit "${YTDLP_RATE}" \
        --no-playlist \
        --continue \
        --no-overwrites \
        --write-info-json \
        --quiet \
        "${URL}" \
        >> "${DOWNLOAD_LOG}" 2>&1; then
        DOWNLOAD_COUNT=$((DOWNLOAD_COUNT + 1))
        info "  ✓ 下载成功: ${VIDEO_ID}"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        warn "  ✗ 下载失败: ${VIDEO_ID} (${URL})"
        echo "${VIDEO_ID} ${URL}" >> "${FAILED_DOWNLOADS}"
    fi

done < "${LONG_VIDEOS_JSONL}"

info "下载汇总: 新下载=${DOWNLOAD_COUNT}, 跳过=${SKIP_COUNT}, 失败=${FAIL_COUNT}"
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    warn "失败列表已保存至: ${FAILED_DOWNLOADS}"
fi

# ---------------------------------------------------------------------------
# 6. 批量建树（main.py index，跳过缓存命中）
# ---------------------------------------------------------------------------

info "=== Step 6: 批量建树 ==="
info "项目根目录: ${PROJECT_ROOT}"
info "配置文件:   ${CONFIG_YAML}"
info "并行进程数: ${WORKERS}"

BUILD_LOG="${LOG_DIR}/build_trees_${TIMESTAMP}.log"
FAILED_BUILDS="${LOG_DIR}/failed_builds_${TIMESTAMP}.txt"
BUILD_COUNT=0
BUILD_SKIP=0
BUILD_FAIL=0

# 构建函数（单个视频）
build_one_video() {
    local video_path="$1"
    local video_stem
    video_stem="$(basename "${video_path%.*}")"
    local cache_file="${TREE_DIR}/${video_stem}_video.pkl"

    # 缓存命中则跳过（pipeline.py 内部也会检查，此处提前判断减少日志噪声）
    if [[ -f "${cache_file}" ]]; then
        echo "[SKIP] ${video_stem}"
        return 0
    fi

    echo "[BUILD] ${video_stem} ← ${video_path}"
    if conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/main.py" \
        index \
        --source "${video_path}" \
        --modality video \
        --config "${CONFIG_YAML}" \
        --env "${ENV_FILE}" \
        >> "${BUILD_LOG}" 2>&1; then
        echo "[OK]   ${video_stem}"
        return 0
    else
        echo "[FAIL] ${video_stem}"
        echo "${video_path}" >> "${FAILED_BUILDS}"
        return 1
    fi
}

export -f build_one_video
export CONDA_ENV PROJECT_ROOT CONFIG_YAML ENV_FILE TREE_DIR BUILD_LOG FAILED_BUILDS

if [[ "${WORKERS}" -gt 1 ]]; then
    # 并行模式：使用 GNU parallel
    if ! command -v parallel &> /dev/null; then
        warn "未找到 GNU parallel，降级为串行模式"
        WORKERS=1
    fi
fi

if [[ "${WORKERS}" -gt 1 ]]; then
    info "并行建树 (jobs=${WORKERS})..."
    find "${VIDEO_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) \
        | parallel -j "${WORKERS}" --bar build_one_video {} \
        2>&1 | tee -a "${MAIN_LOG}"
else
    # 串行模式
    while IFS= read -r -d '' video_path; do
        video_stem="$(basename "${video_path%.*}")"
        cache_file="${TREE_DIR}/${video_stem}_video.pkl"

        if [[ -f "${cache_file}" ]]; then
            BUILD_SKIP=$((BUILD_SKIP + 1))
            continue
        fi

        info "[建树 $((BUILD_COUNT + BUILD_SKIP + 1))/${TOTAL_VIDEOS}] ${video_stem}"
        if conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/main.py" \
            index \
            --source "${video_path}" \
            --modality video \
            --config "${CONFIG_YAML}" \
            --env "${ENV_FILE}" \
            >> "${BUILD_LOG}" 2>&1; then
            BUILD_COUNT=$((BUILD_COUNT + 1))
            info "  ✓ 建树成功: ${video_stem}"
        else
            BUILD_FAIL=$((BUILD_FAIL + 1))
            warn "  ✗ 建树失败: ${video_stem}"
            echo "${video_path}" >> "${FAILED_BUILDS}"
        fi
    done < <(find "${VIDEO_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) -print0)
fi

# ---------------------------------------------------------------------------
# 7. 最终汇总
# ---------------------------------------------------------------------------

info "=== Step 7: 汇总 ==="

TREE_COUNT=$(find "${TREE_DIR}" -name "*_video.pkl" -type f 2>/dev/null | wc -l)
VIDEO_COUNT=$(find "${VIDEO_DIR}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)

info "======================================"
info "  长视频元数据数量:  ${TOTAL_VIDEOS}"
info "  已下载视频数量:    ${VIDEO_COUNT}"
info "  已完成树索引数量:  ${TREE_COUNT}"
info "  本次新建树:        ${BUILD_COUNT}"
info "  跳过（缓存命中）: ${BUILD_SKIP}"
info "  建树失败:          ${BUILD_FAIL}"
info "======================================"
info "主日志:   ${MAIN_LOG}"
info "下载日志: ${DOWNLOAD_LOG}"
info "建树日志: ${BUILD_LOG}"

if [[ ${BUILD_FAIL} -gt 0 ]]; then
    warn "有 ${BUILD_FAIL} 个视频建树失败，详见: ${FAILED_BUILDS}"
    warn "可重新运行脚本以续建失败项（自动跳过已缓存）"
fi

if [[ "${TREE_COUNT}" -ge "${TOTAL_VIDEOS}" ]]; then
    info "✅ 所有长视频树索引已构建完成！"
else
    warn "⚠ 树索引尚未全部完成 (${TREE_COUNT}/${TOTAL_VIDEOS})，可重新运行以续建"
fi

info "脚本完成。"

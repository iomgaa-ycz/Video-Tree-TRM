"""
视频树构建模块
==============
将长视频通过 L2 轴心策略 + VLM 帧描述转化为三层 TreeIndex。

构建策略::

    Step 1: _segment_video  — 固定步长切分，确定 L1 时间边界
    Step 2: L2 先行         — 每个 L2 clip 均匀 seek l2_representative_frames 帧（稀疏），VLM 生成片段描述（1-2句）
    Step 3: L3 向下         — 注入 L2 上下文，VLM 批量帧描述（每帧1-2句）
    Step 4: L1 向上         — 聚合 L2 描述，LLM 生成 L1 摘要（2-3句）
    Step 5: 组装 TreeIndex

L2 轴心策略解决了循环依赖：
    - L2 描述不依赖 L3，从代表帧直接生成
    - L3 注入 L2 上下文后逐帧描述
    - L1 聚合 L2 描述，保证完整覆盖

帧持久化：
    - 帧图像保存到 {cache_dir}/frames/{video_stem}/，长期有效
    - 已提取的帧自动跳过（缓存复用）

使用方式::

    builder = VideoTreeBuilder(embed_model, vlm_client, config.tree)
    index = builder.build("path/to/video.mp4")
    index.save("cache/my_video.pkl")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as cfwait
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger_system import ensure, log_json, log_msg
from video_tree_trm.config import TreeConfig
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.tree_index import (
    IndexMeta,
    L1Node,
    L2Node,
    L3Node,
    TreeIndex,
)

# ---------------------------------------------------------------------------
# Prompt 常量
# ---------------------------------------------------------------------------

_L2_VIDEO_PROMPT = "用1-2句话描述以下视频片段的核心内容，与同级片段形成区分。"

_L3_VIDEO_PROMPT = (
    '该片段的整体内容: "{l2_description}"\n'
    "以下是该片段中连续的 {n} 帧画面。\n"
    "对每帧用一到两句话描述其具体画面内容。\n"
    "重点关注: 动作、物体变化、文字信息、人物表情。\n"
    "不要重复片段整体描述，聚焦每帧的区分性信息。\n"
    '只返回 JSON 数组，格式: ["帧1描述", "帧2描述", ...]，不要其他内容。'
)

_L1_VIDEO_PROMPT = (
    "以下是一个视频段落中各片段的描述:\n{l2_texts}\n"
    "用2-3句话总结该段落的整体内容，涵盖所有片段的主题。"
)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class VideoTreeBuilder:
    """视频模态树构建器。

    将长视频通过 L2 轴心策略（先构建 L2，再向下扩展 L3，向上聚合 L1）
    转化为三层 TreeIndex。所有描述通过 VLM 生成，节点 embedding 均为 None（由 Pipeline.embed_all 延迟填充）。

    属性:
        vlm: VLM/LLM 客户端（用于图文和纯文本调用）。
        config: 树构建配置。
    """

    def __init__(
        self,
        vlm: LLMClient,
        config: TreeConfig,
    ) -> None:
        """初始化视频树构建器。

        参数:
            vlm: 已初始化的 VLM/LLM 客户端（LLMClient），需支持 chat_with_images。
            config: 树构建配置（TreeConfig），关键字段：
                    l1_segment_duration, l2_clip_duration, l3_fps,
                    l2_representative_frames, cache_dir。

        实现细节:
            构建器不持有 EmbeddingModel，所有 embedding 延迟到检索阶段由 Pipeline 统一计算。
        """
        self.vlm = vlm
        self.config = config

    # ------------------------------------------------------------------
    # URL 流式辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _is_url(path_or_url: str) -> bool:
        """判断输入是否为网络 URL（而非本地路径）。

        参数:
            path_or_url: 文件路径或 URL 字符串。

        返回:
            True 表示 URL，False 表示本地路径。
        """
        return path_or_url.startswith(("http://", "https://"))

    @staticmethod
    def _source_stem(video_path: str) -> str:
        """从视频路径或 YouTube URL 中提取短标识符，用于帧缓存目录命名。

        参数:
            video_path: 本地文件路径或 YouTube 视频页面 URL。

        返回:
            短字符串标识符（本地文件取 stem，YouTube URL 取 v= 后的视频 ID）。

        实现细节:
            YouTube URL 格式: https://www.youtube.com/watch?v=VIDEO_ID
            CDN 直链（stream_url）不应传入此方法，应传入原始 video_path。
        """
        if "youtube.com/watch" in video_path or "youtu.be/" in video_path:
            # 提取 v= 参数
            match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{8,15})", video_path)
            if match:
                return match.group(1)
        # 本地文件或其他 URL 取文件名 stem，限制最大长度避免文件名过长
        stem = Path(video_path).stem
        return stem[:64] if len(stem) > 64 else stem

    @staticmethod
    def _resolve_stream(url: str) -> str:
        """通过 yt-dlp 获取 YouTube 视频的 CDN 直链，供 cv2.VideoCapture 直接使用。

        参数:
            url: YouTube 视频页面 URL。

        返回:
            CDN HTTPS 直链（ffmpeg/OpenCV 可直接流式读取）。

        实现细节:
            调用 yt-dlp -g 获取最佳 mp4 格式直链，仅请求元数据不下载内容。
        """
        log_msg("INFO", "获取 YouTube CDN 直链", url=url)
        result = subprocess.run(
            [
                "yt-dlp",
                "-g",
                "--format",
                "best[ext=mp4][height<=720]/best[ext=mp4]/best",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ensure(result.returncode == 0, f"yt-dlp 获取直链失败: {result.stderr.strip()}")
        stream_url = result.stdout.strip().splitlines()[0]
        ensure(stream_url.startswith("http"), f"yt-dlp 返回非 URL: {stream_url[:100]}")
        log_msg("INFO", "CDN 直链获取成功", stream_url=stream_url[:80])
        return stream_url

    @staticmethod
    def _get_video_duration(url: str) -> float:
        """通过 yt-dlp --dump-json 获取视频时长（秒）。

        参数:
            url: YouTube 视频页面 URL。

        返回:
            视频总时长（秒，浮点数）。

        实现细节:
            对 HTTP 流 cv2.CAP_PROP_FRAME_COUNT 常返回 -1 或 0，
            使用 yt-dlp 元数据获取准确时长。
        """
        log_msg("INFO", "获取视频时长元数据", url=url)
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ensure(result.returncode == 0, f"yt-dlp 元数据获取失败: {result.stderr.strip()}")
        meta = json.loads(result.stdout)
        duration = float(meta.get("duration", 0))
        ensure(duration > 0, f"视频时长读取异常: {duration}")
        log_msg("INFO", "视频时长确认", duration_sec=round(duration, 1))
        return duration

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build(self, video_path: str) -> TreeIndex:
        """将长视频构建为三层 TreeIndex（异步事件循环模式）。

        参数:
            video_path: 视频文件路径（.mp4/.avi/.mkv 等）或 YouTube URL。
                        传入 URL 时通过 yt-dlp 获取 CDN 直链后流式读取，无需下载。

        返回:
            三层 TreeIndex 对象。

        实现细节:
            采用事件循环 + Future 链式触发（非阻塞）：
            Step 1: _segment_video → L1 区间列表
            Step 2: 收集所有 L2 任务，预计算每个 L1 的 L2 数量
            Step 3: 一次性提交所有 L2 任务（非阻塞，max_workers=concurrency）
            Step 4: 事件循环（cfwait FIRST_COMPLETED）：
                    L2 完成 → 立即提交 L3 任务
                    L3 完成 → 检查 L1 就绪 → 立即提交 L1 任务
                    L1 完成 → 收集结果
            Step 5: 有序重建 l1_nodes，组装 TreeIndex
            主线程单线程操作 l1_l2_buckets，无竞争，无需 Lock。
        """
        # Phase 0: URL vs 本地文件处理
        if self._is_url(video_path):
            stream_url = self._resolve_stream(video_path)
            duration_hint: Optional[float] = self._get_video_duration(video_path)
            log_msg("INFO", "开始构建视频树索引（URL 流式模式）", source_url=video_path)
        else:
            ensure(os.path.isfile(video_path), f"视频文件不存在: {video_path}")
            stream_url = video_path
            duration_hint = None
            log_msg("INFO", "开始构建视频树索引", video_path=video_path)

        source_id = self._source_stem(video_path)

        # Phase 1: 时间切分
        l1_ranges = self._segment_video(stream_url, duration_hint=duration_hint)
        ensure(len(l1_ranges) > 0, "视频时间切分结果为空")
        log_msg("INFO", "视频切分完成", l1_count=len(l1_ranges))

        # Phase 2: 收集全局 L2 任务列表 + 预计算每个 L1 的 L2 数量
        all_l2_tasks: List[Tuple[int, int, str, Tuple[float, float]]] = []
        l2_counts: Dict[int, int] = {}
        for i, l1_range in enumerate(l1_ranges):
            clips = self._get_l2_clips(l1_range)
            l2_counts[i] = len(clips)
            for j, clip_range in enumerate(clips):
                all_l2_tasks.append((i, j, f"l1_{i}_l2_{j}", clip_range))

        # Phase 3 & 4: 异步事件循环（ThreadPoolExecutor + cfwait FIRST_COMPLETED）
        # pending: Future → ("l2"|"l3"|"l1", *meta)
        pending: Dict[Future, tuple] = {}
        l1_l2_buckets: Dict[int, Dict[int, L2Node]] = {i: {} for i in range(len(l1_ranges))}
        l1_nodes_result: Dict[int, L1Node] = {}

        pool = ThreadPoolExecutor(max_workers=self.config.concurrency)

        # 一次性提交所有 L2 任务（非阻塞）
        for l1_i, l2_j, l2_id, clip_range in all_l2_tasks:
            fut = pool.submit(
                self._build_l2_video, stream_url, clip_range, l2_id, source_id
            )
            pending[fut] = ("l2", l1_i, l2_j, clip_range)

        log_msg(
            "INFO",
            "已提交全部 L2 任务（异步）",
            total_l2=len(all_l2_tasks),
            concurrency=self.config.concurrency,
        )

        # 事件循环：监听完成 → 触发下游
        while pending:
            done, _ = cfwait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                kind, *meta = pending.pop(fut)

                if kind == "l2":
                    l1_i, l2_j, clip_range = meta
                    l2_node: L2Node = fut.result()
                    # L2 完成 → 立即提交 L3 任务（非阻塞）
                    l3_fut = pool.submit(
                        self._build_l3_task,
                        stream_url, l2_node, clip_range, source_id, l1_i, l2_j,
                    )
                    pending[l3_fut] = ("l3", l1_i, l2_j)
                    log_msg(
                        "INFO", "L2 VLM 完成，已触发 L3 任务",
                        l2_id=f"l1_{l1_i}_l2_{l2_j}",
                    )

                elif kind == "l3":
                    l1_i, l2_j = meta
                    completed_l2: L2Node = fut.result()  # L2Node（含 children）
                    l1_l2_buckets[l1_i][l2_j] = completed_l2
                    log_msg(
                        "INFO", "L3 完成",
                        l2_id=f"l1_{l1_i}_l2_{l2_j}",
                        l3_count=len(completed_l2.children),
                    )
                    # 检查该 L1 的所有 L2 是否就绪 → 触发 L1
                    if len(l1_l2_buckets[l1_i]) == l2_counts[l1_i]:
                        ordered_l2 = [
                            l1_l2_buckets[l1_i][j] for j in range(l2_counts[l1_i])
                        ]
                        l1_fut = pool.submit(
                            self._build_l1_video,
                            ordered_l2, f"l1_{l1_i}", l1_ranges[l1_i],
                        )
                        pending[l1_fut] = ("l1", l1_i)
                        log_msg("INFO", "L1 触发", l1_id=f"l1_{l1_i}")

                elif kind == "l1":
                    (l1_i,) = meta
                    l1_nodes_result[l1_i] = fut.result()
                    log_msg(
                        "INFO", "L1 节点构建完成",
                        l1_id=f"l1_{l1_i}",
                        l2_count=l2_counts[l1_i],
                    )

        pool.shutdown(wait=False)

        # Phase 5: 有序重建 l1_nodes，组装 TreeIndex
        l1_nodes: List[L1Node] = [l1_nodes_result[i] for i in range(len(l1_ranges))]

        metadata = IndexMeta(
            source_path=video_path,
            modality="video",
            created_at=datetime.now().isoformat(),
        )
        index = TreeIndex(metadata=metadata, roots=l1_nodes)

        total_l2 = sum(len(r.children) for r in l1_nodes)
        total_l3 = sum(len(l2.children) for r in l1_nodes for l2 in r.children)
        log_json(
            "video_tree_build",
            {
                "source_path": video_path,
                "l1_count": len(l1_nodes),
                "l2_count": total_l2,
                "l3_count": total_l3,
                "embedded": False,
            },
        )
        log_msg(
            "INFO",
            "视频树索引构建完成",
            l1=len(l1_nodes),
            l2=total_l2,
            l3=total_l3,
        )
        return index

    # ------------------------------------------------------------------
    # 内部方法：时间切分
    # ------------------------------------------------------------------

    def _segment_video(
        self,
        video_path: str,
        duration_hint: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """读取视频总时长，按固定步长切分为 L1 时间区间列表。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            duration_hint: 已知视频时长（秒）。传入时跳过 cv2 读取，
                           用于 HTTP 流（CAP_PROP_FRAME_COUNT 在流上返回 -1）。

        返回:
            L1 时间区间列表，每项为 (start_sec, end_sec)。
            最后一段可能短于 l1_segment_duration。

        实现细节:
            本地文件: 使用 cv2.VideoCapture 读取总帧数和 FPS 计算总时长。
            HTTP 流: 使用 duration_hint（由 yt-dlp --dump-json 获取），
                    避免 CAP_PROP_FRAME_COUNT 在流上不可靠的问题。
        """
        if duration_hint is not None:
            total_duration = duration_hint
        else:
            cap = cv2.VideoCapture(video_path)
            ensure(cap.isOpened(), f"无法打开视频文件: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            ensure(fps > 0, f"视频 FPS 读取异常: {fps}")
            ensure(total_frames > 0, f"视频总帧数读取异常: {total_frames}")
            total_duration = total_frames / fps
        step = self.config.l1_segment_duration
        ranges: List[Tuple[float, float]] = []

        start = 0.0
        while start < total_duration:
            end = min(start + step, total_duration)
            ranges.append((start, end))
            start = end

        log_msg(
            "INFO",
            "L1 时间切分",
            total_duration=round(total_duration, 2),
            l1_count=len(ranges),
        )
        return ranges

    def _get_l2_clips(self, l1_range: Tuple[float, float]) -> List[Tuple[float, float]]:
        """将 L1 时间区间等分为 L2 clips。

        参数:
            l1_range: L1 时间区间 (start, end)，单位秒。

        返回:
            L2 clip 时间区间列表，每项为 (start, end)。
            最后一段可能短于 l2_clip_duration。
        """
        start, end = l1_range
        step = self.config.l2_clip_duration
        clips: List[Tuple[float, float]] = []

        t = start
        while t < end:
            clip_end = min(t + step, end)
            clips.append((t, clip_end))
            t = clip_end

        return clips

    # ------------------------------------------------------------------
    # 内部方法：帧提取
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: str,
        time_range: Tuple[float, float],
        fps: float,
        source_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """按指定 fps 提取时间范围内的帧，保存到 cache 目录。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            time_range: 提取时间区间 (start_sec, end_sec)。
            fps: 提取帧率（帧/秒）。
            source_id: 帧缓存目录名（若为 None 则从 video_path 提取 stem）。
                       传入 URL 时应显式提供（避免文件名过长）。

        返回:
            [(frame_path, timestamp_sec), ...]，按时间顺序排列。
            已存在的帧文件直接复用，不重复提取。

        实现细节:
            帧保存路径: {cache_dir}/frames/{source_id}/{start:.1f}_{ts:.3f}.jpg
            使用 cv2.VideoCapture.set(CAP_PROP_POS_MSEC) 精确定位帧位置。
        """
        video_stem = source_id if source_id is not None else self._source_stem(video_path)
        frame_dir = Path(self.config.cache_dir) / "frames" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        start_sec, end_sec = time_range
        step = 1.0 / fps  # 每帧时间间隔（秒）

        timestamps: List[float] = []
        t = start_sec
        while t < end_sec:
            timestamps.append(t)
            t += step

        if not timestamps:
            log_msg(
                "WARNING",
                "帧提取时间区间内无有效时间戳",
                time_range=time_range,
                fps=fps,
            )
            return []

        result: List[Tuple[str, float]] = []
        cap: Optional[cv2.VideoCapture] = None

        for ts in timestamps:
            frame_name = f"{start_sec:.1f}_{ts:.3f}.jpg"
            frame_path = str(frame_dir / frame_name)

            # 缓存复用：文件已存在则跳过提取
            if os.path.isfile(frame_path):
                result.append((frame_path, ts))
                continue

            # 延迟打开 VideoCapture
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                ensure(cap.isOpened(), f"无法打开视频: {video_path}")

            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret:
                log_msg(
                    "WARNING",
                    "帧读取失败，跳过",
                    timestamp=ts,
                    video_path=video_path,
                )
                continue

            cv2.imwrite(frame_path, frame)
            result.append((frame_path, ts))

        if cap is not None:
            cap.release()

        return result

    # ------------------------------------------------------------------
    # 内部方法：节点构建
    # ------------------------------------------------------------------

    def _build_l2_video(
        self,
        video_path: str,
        clip_range: Tuple[float, float],
        l2_id: str,
        source_id: Optional[str] = None,
    ) -> L2Node:
        """构建 L2 视频节点（代表帧 VLM 描述 + 嵌入）。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            clip_range: L2 clip 时间区间 (start, end)，单位秒。
            l2_id: 节点 ID。
            source_id: 帧缓存目录名（若为 None 则从 video_path 提取 stem）。

        返回:
            L2Node（children 为空，由调用方填充）。

        实现细节:
            1. 在 clip 时间区间内均匀计算 l2_representative_frames 个时间戳（稀疏采样）
            2. 直接 seek 到各时间戳提取帧（独立于 l3_fps，不走全量提取路径）
            3. 将代表帧路径传入 vlm.chat_with_images 生成 1-2 句描述
            4. 对描述文本调用 embed 获取嵌入向量
        """
        start_sec, end_sec = clip_range
        n_rep = self.config.l2_representative_frames
        # 均匀计算 n_rep 个时间戳（首尾均包含）
        if n_rep == 1:
            timestamps = [(start_sec + end_sec) / 2.0]
        else:
            step = (end_sec - start_sec) / (n_rep - 1)
            timestamps = [start_sec + i * step for i in range(n_rep)]

        video_stem = source_id if source_id is not None else self._source_stem(video_path)
        frame_dir = Path(self.config.cache_dir) / "frames" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        rep_frames: List[str] = []
        cap = cv2.VideoCapture(video_path)
        ensure(cap.isOpened(), f"无法打开视频: {video_path}")
        for ts in timestamps:
            frame_name = f"l2_{ts:.3f}.jpg"
            frame_path = str(frame_dir / frame_name)
            if not os.path.isfile(frame_path):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    log_msg("WARNING", "L2 代表帧读取失败，跳过", timestamp=ts)
                    continue
                cv2.imwrite(frame_path, frame)
            rep_frames.append(frame_path)
        cap.release()

        ensure(len(rep_frames) > 0, f"L2 节点 {l2_id} 代表帧提取结果为空")

        description = self.vlm.chat_with_images(_L2_VIDEO_PROMPT, images=rep_frames)
        return L2Node(
            id=l2_id,
            description=description,
            embedding=None,
            time_range=clip_range,
        )

    def _build_l3_video(
        self,
        frames: List[Tuple[str, float]],
        l2_description: str,
        l1_i: int,
        l2_j: int,
    ) -> List[L3Node]:
        """批量构建 L3 视频节点（注入 L2 上下文的 VLM 帧描述 + 嵌入）。

        参数:
            frames: [(frame_path, timestamp), ...]，来自 _extract_frames。
            l2_description: L2 节点描述，作为上下文注入 prompt。
            l1_i: 父 L1 索引（用于生成节点 ID）。
            l2_j: 父 L2 索引（用于生成节点 ID）。

        返回:
            L3Node 列表，每项对应一帧。

        实现细节:
            1. 将全部帧路径和 L2 上下文构建批量 prompt，一次 VLM 调用
            2. 要求 VLM 返回 JSON 数组（每项为一帧的描述）
            3. 若 JSON 解析失败，降级为逐帧单次调用（不抛异常）
            4. 描述文本批量嵌入（一次 embed 调用）
        """
        ensure(len(frames) > 0, f"L3 帧列表为空 (l1={l1_i}, l2={l2_j})")

        frame_paths = [fp for fp, _ in frames]
        n = len(frame_paths)

        prompt = _L3_VIDEO_PROMPT.format(
            l2_description=l2_description,
            n=n,
        )

        # Phase 1: 尝试批量调用 VLM
        descriptions = self._call_vlm_batch(prompt, frame_paths, n, l1_i, l2_j)

        # Phase 2: 构建 L3 节点（embedding=None，延迟到 embed_all）
        nodes: List[L3Node] = []
        for k, (desc, (frame_path, ts)) in enumerate(zip(descriptions, frames)):
            nodes.append(
                L3Node(
                    id=f"l1_{l1_i}_l2_{l2_j}_l3_{k}",
                    description=desc,
                    embedding=None,
                    raw_content=None,
                    frame_path=frame_path,
                    timestamp=ts,
                )
            )
        return nodes

    def _build_l3_task(
        self,
        video_path: str,
        l2_node: L2Node,
        clip_range: Tuple[float, float],
        source_id: str,
        l1_i: int,
        l2_j: int,
    ) -> L2Node:
        """L3 线程任务：提取帧 + 批量 VLM 帧描述，将 l3_nodes 赋给 l2_node.children。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            l2_node: 已构建的 L2 节点（description 已就绪）。
            clip_range: L2 clip 时间区间 (start, end)，单位秒。
            source_id: 帧缓存目录名。
            l1_i: 父 L1 索引（用于 L3 节点 ID 生成）。
            l2_j: 父 L2 索引（用于 L3 节点 ID 生成）。

        返回:
            已填充 children 的 L2Node（供事件循环收集）。

        实现细节:
            作为独立线程任务单元，内部独立持有 VideoCapture，线程安全。
            由 build() 事件循环在 L2 任务完成后自动提交（非阻塞）。
        """
        all_frames = self._extract_frames(
            video_path, clip_range, self.config.l3_fps, source_id=source_id
        )
        l3_nodes = self._build_l3_video(all_frames, l2_node.description, l1_i, l2_j)
        l2_node.children = l3_nodes
        return l2_node

    def _call_vlm_batch(
        self,
        prompt: str,
        frame_paths: List[str],
        n: int,
        l1_i: int,
        l2_j: int,
    ) -> List[str]:
        """尝试批量 VLM 调用，解析失败时降级为逐帧调用。

        参数:
            prompt: 注入 L2 上下文的批量描述 prompt。
            frame_paths: 帧图像路径列表。
            n: 帧数。
            l1_i: 父 L1 索引（用于日志）。
            l2_j: 父 L2 索引（用于日志）。

        返回:
            与 frame_paths 等长的描述文本列表。
        """
        # 尝试批量调用
        try:
            raw = self.vlm.chat_with_images(prompt, images=frame_paths)
            descriptions = self._parse_json_descriptions(raw, n)
            if descriptions is not None:
                return descriptions
            log_msg(
                "WARNING",
                "L3 批量 VLM 返回 JSON 解析失败，降级逐帧调用",
                l1=l1_i,
                l2=l2_j,
                raw_preview=raw[:100],
            )
        except Exception as exc:
            log_msg(
                "WARNING",
                f"L3 批量 VLM 调用异常，降级逐帧调用: {exc}",
                l1=l1_i,
                l2=l2_j,
            )

        # 降级：逐帧调用
        single_prompt = (
            f'该片段的整体内容: "{prompt.split(chr(10))[0]}"\n'
            "用一到两句话描述这帧画面的具体内容。"
            "重点关注: 动作、物体变化、文字信息、人物表情。"
        )
        return [
            self.vlm.chat_with_images(single_prompt, images=[fp]) for fp in frame_paths
        ]

    def _parse_json_descriptions(
        self, raw: str, expected_n: int
    ) -> Optional[List[str]]:
        """从 VLM 输出中解析 JSON 描述数组。

        参数:
            raw: VLM 原始返回字符串。
            expected_n: 期望的描述条数。

        返回:
            成功解析且长度匹配时返回 List[str]，否则返回 None。
        """
        raw = raw.strip()
        # 提取可能被 markdown 代码块包裹的 JSON
        code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if code_match:
            raw = code_match.group(1)

        if not raw.startswith("["):
            return None

        try:
            items: List[str] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(items, list) or len(items) != expected_n:
            return None

        return [str(item).strip() for item in items]

    def _build_l1_video(
        self,
        l2_children: List[L2Node],
        l1_id: str,
        l1_range: Tuple[float, float],
    ) -> L1Node:
        """聚合 L2 描述，构建 L1 节点（LLM 摘要 + 嵌入）。

        参数:
            l2_children: 该 L1 节点下的所有 L2 节点。
            l1_id: 节点 ID。
            l1_range: L1 时间区间 (start, end)，单位秒。

        返回:
            L1Node（children 已赋值）。

        实现细节:
            将所有 L2 描述拼接后送入 vlm.chat()（纯文本），
            生成 2-3 句覆盖所有 L2 语义的摘要。
        """
        ensure(len(l2_children) > 0, f"L1 节点 {l1_id} 没有 L2 子节点")
        l2_texts = "\n".join(f"- {node.description}" for node in l2_children)
        prompt = _L1_VIDEO_PROMPT.format(l2_texts=l2_texts)
        summary = self.vlm.chat(prompt)
        return L1Node(
            id=l1_id,
            summary=summary,
            embedding=None,
            time_range=l1_range,
            children=l2_children,
        )

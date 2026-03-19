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

并发模型（异步版）::

    build() → asyncio.run(_build_async())
    _build_async():
        asyncio.Semaphore(concurrency=16) 控制最大 VLM 并发数
        Phase 1: asyncio.gather(所有L2任务)  — 16路同时 VLM
        Phase 2: asyncio.gather(所有L3任务)  — 每个L3任务内的12批次同时并发
        Phase 3: asyncio.gather(各L1摘要)    — L1收齐后并发
    ffmpeg 提帧在 ThreadPoolExecutor(max_workers=8) 中并行执行

L2 轴心策略解决了循环依赖：
    - L2 描述不依赖 L3，从代表帧直接生成
    - L3 注入 L2 上下文后逐帧描述
    - L1 聚合 L2 描述，保证完整覆盖

帧持久化：
    - 帧图像保存到 {cache_dir}/frames/{video_stem}/，长期有效
    - 已提取的帧自动跳过（缓存复用）

使用方式::

    builder = VideoTreeBuilder(vlm_client, config.tree)
    index = builder.build("path/to/video.mp4")   # 同步壳，内部 asyncio.run()
    index.save("cache/my_video.pkl")
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from utils.logger_system import ensure, log_json, log_msg
from video_tree_trm.config import TreeConfig
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.tree_index import (
    IndexMeta,
    L1Node,
    L2Node,
    L3Node,
    TreeIndex,
    load_l1_json,
    save_l1_json,
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

# 每次 VLM 调用携带的最大帧数：5 帧 payload 小、JSON 解析成功率高
_L3_BATCH_SIZE = 5

_L3_SINGLE_PROMPT = (
    '该片段的整体内容: "{l2_description}"\n'
    "用一到两句话描述这帧画面的具体内容。"
    "重点关注: 动作、物体变化、文字信息、人物表情。"
)

# ffmpeg 并发提帧的线程池大小（CPU 密集型，避免过度并发）
_FFMPEG_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class VideoTreeBuilder:
    """视频模态树构建器（asyncio 真并发版）。

    将长视频通过 L2 轴心策略（先构建 L2，再向下扩展 L3，向上聚合 L1）
    转化为三层 TreeIndex。

    并发架构:
        build() 为同步壳，内部调用 asyncio.run(_build_async())。
        _build_async() 使用 asyncio.Semaphore(concurrency) 控制并发 VLM 数量。
        所有 VLM 调用通过 LLMClient 的异步接口（AsyncOpenAI）发起，零线程阻塞。
        ffmpeg 提帧在独立 ThreadPoolExecutor 中并行，不阻塞事件循环。

    属性:
        vlm: VLM/LLM 客户端（用于图文和纯文本调用）。
        config: 树构建配置。
        _ffmpeg_pool: ffmpeg 专用线程池（max_workers=_FFMPEG_MAX_WORKERS）。
    """

    def __init__(
        self,
        vlm: LLMClient,
        config: TreeConfig,
    ) -> None:
        """初始化视频树构建器。

        参数:
            vlm: 已初始化的 VLM/LLM 客户端（LLMClient），需支持异步接口。
            config: 树构建配置（TreeConfig），关键字段：
                    l1_segment_duration, l2_clip_duration, l3_fps,
                    l2_representative_frames, cache_dir, concurrency。

        实现细节:
            ffmpeg 线程池在构建器级别创建，所有异步协程共用，
            避免每次提帧都重建线程池的开销。
        """
        self.vlm = vlm
        self.config = config
        self._ffmpeg_pool = ThreadPoolExecutor(max_workers=_FFMPEG_MAX_WORKERS)
        # 进度与中间结果目录均挂在 cache_dir 下，避免散落其它位置
        self._cache_root = Path(self.config.cache_dir)

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
        """
        if "youtube.com/watch" in video_path or "youtu.be/" in video_path:
            match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{8,15})", video_path)
            if match:
                return match.group(1)
        stem = Path(video_path).stem
        return stem[:64] if len(stem) > 64 else stem

    @staticmethod
    def _resolve_stream(url: str) -> str:
        """通过 yt-dlp 获取 YouTube 视频的 CDN 直链，供 cv2.VideoCapture 直接使用。

        参数:
            url: YouTube 视频页面 URL。

        返回:
            CDN HTTPS 直链（ffmpeg/OpenCV 可直接流式读取）。
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
        """将长视频构建为三层 TreeIndex（同步壳，内部 asyncio.run 驱动）。

        参数:
            video_path: 视频文件路径（.mp4/.avi/.mkv 等）或 YouTube URL。

        返回:
            三层 TreeIndex 对象。

        实现细节:
            同步壳设计保持与 build_trees_batch.py 的接口兼容性。
            每次调用 asyncio.run() 创建独立事件循环，多线程安全（各线程独立循环）。
        """
        return asyncio.run(self._build_async(video_path))

    # ------------------------------------------------------------------
    # 核心异步构建逻辑
    # ------------------------------------------------------------------

    async def _build_async(self, video_path: str) -> TreeIndex:
        """异步构建三层 TreeIndex（真并发核心，L2→L3 链式触发）。

        参数:
            video_path: 视频文件路径或 YouTube URL。

        返回:
            三层 TreeIndex 对象。

        实现细节:
            并发架构：每个 L1 段内启动一组"L2→L3 链式协程"，
            L2 完成后立即触发 L3（不等待其他 L2），L3 完成后触发 L1 摘要。
            各 L1 段独立并发，彼此不阻塞。
            Semaphore(concurrency=16) 全局限制同时在途 VLM 调用数量。

            关键调用链（每个 L2 clip 独立）::
                _build_segment(i) → asyncio.gather(
                    _chain(i,0): _build_l2_video_async → _build_l3_task_async
                    _chain(i,1): _build_l2_video_async → _build_l3_task_async
                    ...
                ) → _build_l1_video_async(i)
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

        # Phase 1: 时间切分（同步，仅一次）
        l1_ranges = self._segment_video(stream_url, duration_hint=duration_hint)
        ensure(len(l1_ranges) > 0, "视频时间切分结果为空")
        log_msg("INFO", "视频切分完成", l1_count=len(l1_ranges))

        total_l1 = len(l1_ranges)

        # Phase 1.1: 读取已有进度（支持断点续跑）
        finished_l1_ids: set[int] = set()
        progress = self._load_progress(source_id)
        if progress is not None and progress.get("total_l1") == total_l1:
            finished_l1_ids = set(progress.get("finished_l1_ids", []))
            if finished_l1_ids:
                log_msg(
                    "INFO",
                    "检测到中间进度，启用断点续跑",
                    stem=source_id,
                    finished_l1=list(sorted(finished_l1_ids)),
                )
        else:
            # 进度不存在或形状不匹配时，从零开始，旧进度视为无效
            if progress is not None:
                log_msg(
                    "WARNING",
                    "进度文件与当前 L1 段数不一致，忽略旧进度",
                    stem=source_id,
                    recorded_total_l1=progress.get("total_l1"),
                    current_total_l1=total_l1,
                )

        # 创建 VLM 并发控制信号量（每视频独立，限制同时在途 VLM 请求数）
        vlm_sem = asyncio.Semaphore(self.config.concurrency)

        # Phase 2-5: 按 L1 段并发，段内 L2→L3 链式触发，L3 收齐后触发 L1
        async def _build_segment(i: int, l1_range: Tuple[float, float]) -> L1Node:
            """单个 L1 段的完整构建：L2+L3 并发链式 → L1 摘要。

            参数:
                i: L1 段索引。
                l1_range: L1 时间区间 (start, end)。

            返回:
                完整的 L1Node（含所有 L2 和 L3 子节点）。

            实现细节:
                段内所有 L2 clip 同时启动（asyncio.gather），
                每个 clip 的 L2 VLM 完成后立即触发 L3，不等待其他 clip 的 L2。
                所有 clip 的 L2+L3 完成后，触发 L1 文本摘要。
            """
            clips = self._get_l2_clips(l1_range)

            async def _chain(j: int, clip_range: Tuple[float, float]) -> Tuple[int, L2Node]:
                """L2→L3 链：L2 完成立即触发 L3，返回 (j, 含children的L2Node)。"""
                l2_id = f"l1_{i}_l2_{j}"
                l2_node = await self._build_l2_video_async(
                    stream_url, clip_range, l2_id, source_id, vlm_sem
                )
                log_msg("INFO", "L2 VLM 完成，已触发 L3 任务", l2_id=l2_id)

                completed_l2 = await self._build_l3_task_async(
                    stream_url, l2_node, clip_range, source_id, i, j, vlm_sem
                )
                log_msg(
                    "INFO", "L3 完成",
                    l2_id=l2_id,
                    l3_count=len(completed_l2.children),
                )
                return (j, completed_l2)

            # 所有 clip 同时启动（不等 L2 全部结束再开 L3）
            pairs = await asyncio.gather(*[_chain(j, clip) for j, clip in enumerate(clips)])
            ordered_l2 = [p[1] for p in sorted(pairs, key=lambda x: x[0])]

            log_msg("INFO", "L1 触发", l1_id=f"l1_{i}")
            l1_node = await self._build_l1_video_async(
                ordered_l2, f"l1_{i}", l1_range, vlm_sem
            )
            log_msg(
                "INFO", "L1 节点构建完成",
                l1_id=f"l1_{i}",
                l2_count=len(ordered_l2),
            )
            return l1_node

        total_clips = sum(len(self._get_l2_clips(r)) for r in l1_ranges)
        log_msg(
            "INFO",
            "开始并发构建（L2→L3链式，L1段间并发，支持断点续跑）",
            total_l2=total_clips,
            concurrency=self.config.concurrency,
        )

        # Phase 2: 并发构建尚未完成的 L1 段（段内 L2+L3 链式并发）
        tasks: List[asyncio.Task[L1Node]] = []
        task_indices: List[int] = []
        for i, r in enumerate(l1_ranges):
            # 已完成且中间 JSON 存在 → 直接复用
            if i in finished_l1_ids and self._has_l1_intermediate(source_id, i):
                continue
            tasks.append(asyncio.create_task(_build_segment(i, r)))
            task_indices.append(i)

        new_l1_nodes: Dict[int, L1Node] = {}
        if tasks:
            results = await asyncio.gather(*tasks)
            for idx, node in zip(task_indices, results):
                # 每完成一个 L1 段就写入中间 JSON，并刷新进度文件
                self._save_l1_intermediate(source_id, node, idx)
                finished_l1_ids.add(idx)
                new_l1_nodes[idx] = node
            self._save_progress(source_id, total_l1, finished_l1_ids)

        # Phase 3: 汇总所有 L1 段（中间 + 新生成）
        l1_nodes: List[L1Node] = []
        for i in range(total_l1):
            if i in new_l1_nodes:
                l1_nodes.append(new_l1_nodes[i])
                continue
            node = self._load_l1_intermediate(source_id, i)
            ensure(node is not None, f"L1 段 {i} 缺失中间结果，无法恢复")
            l1_nodes.append(node)

        # Phase 6: 组装 TreeIndex
        metadata = IndexMeta(
            source_path=video_path,
            modality="video",
            created_at=datetime.now().isoformat(),
        )
        index = TreeIndex(metadata=metadata, roots=l1_nodes)

        total_l2_count = sum(len(r.children) for r in l1_nodes)
        total_l3_count = sum(len(l2.children) for r in l1_nodes for l2 in r.children)
        log_json(
            "video_tree_build",
            {
                "source_path": video_path,
                "l1_count": len(l1_nodes),
                "l2_count": total_l2_count,
                "l3_count": total_l3_count,
                "embedded": False,
            },
        )
        log_msg(
            "INFO",
            "视频树索引构建完成",
            l1=len(l1_nodes),
            l2=total_l2_count,
            l3=total_l3_count,
        )
        # 最终 JSON 写入成功后，清理由断点机制生成的中间文件
        self._cleanup_intermediate_and_progress(source_id)
        return index

    # ------------------------------------------------------------------
    # 内部方法：时间切分（同步，仅执行一次）
    # ------------------------------------------------------------------

    def _segment_video(
        self,
        video_path: str,
        duration_hint: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """读取视频总时长，按固定步长切分为 L1 时间区间列表。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            duration_hint: 已知视频时长（秒）。传入时跳过 cv2 读取。

        返回:
            L1 时间区间列表，每项为 (start_sec, end_sec)。
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
            L2 clip 时间区间列表。
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
    # 内部方法：帧提取（ffmpeg subprocess，在线程池执行）
    # ------------------------------------------------------------------

    def _ffmpeg_extract_frame(self, video_path: str, ts: float, out_path: str) -> bool:
        """用 ffmpeg subprocess 提取单帧图像，兼容 AV1/H.264 等所有编码格式。

        参数:
            video_path: 视频文件路径（本地 MP4 或 CDN URL）。
            ts: 目标时间戳（秒）。
            out_path: 输出 JPEG 文件路径。

        返回:
            True 表示提取成功，False 表示失败。
        """
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-y", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and os.path.isfile(out_path)

    async def _extract_frames_async(
        self,
        video_path: str,
        time_range: Tuple[float, float],
        fps: float,
        source_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """异步并发提取时间范围内的帧，保存到 cache 目录。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            time_range: 提取时间区间 (start_sec, end_sec)。
            fps: 提取帧率（帧/秒）。
            source_id: 帧缓存目录名。

        返回:
            [(frame_path, timestamp_sec), ...]，按时间顺序排列。

        实现细节:
            所有 ffmpeg 提取任务通过 run_in_executor(self._ffmpeg_pool, ...) 并发执行，
            已缓存的帧直接跳过（无需调用 ffmpeg）。
            ffmpeg 线程池 max_workers=_FFMPEG_MAX_WORKERS 防止过度并发占满 CPU。
        """
        video_stem = source_id if source_id is not None else self._source_stem(video_path)
        frame_dir = Path(self.config.cache_dir) / "frames" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        start_sec, end_sec = time_range
        step = 1.0 / fps

        timestamps: List[float] = []
        t = start_sec
        while t < end_sec:
            timestamps.append(t)
            t += step

        if not timestamps:
            log_msg("WARNING", "帧提取时间区间内无有效时间戳", time_range=time_range, fps=fps)
            return []

        loop = asyncio.get_event_loop()

        async def _extract_one(ts: float) -> Optional[Tuple[str, float]]:
            """提取单帧：缓存命中直接返回，否则在线程池中调用 ffmpeg。"""
            frame_name = f"{start_sec:.1f}_{ts:.3f}.jpg"
            frame_path = str(frame_dir / frame_name)

            if os.path.isfile(frame_path):
                return (frame_path, ts)

            success = await loop.run_in_executor(
                self._ffmpeg_pool,
                self._ffmpeg_extract_frame, video_path, ts, frame_path,
            )
            if not success:
                log_msg("WARNING", "帧读取失败，跳过", timestamp=ts, video_path=video_path)
                return None
            return (frame_path, ts)

        # 并发提取所有帧（受 ffmpeg 线程池限制，不会无限并发）
        results = await asyncio.gather(*[_extract_one(ts) for ts in timestamps])
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # 内部方法：L1 中间结果与进度管理（断点续跑）
    # ------------------------------------------------------------------

    def _intermediate_dir(self, stem: str) -> Path:
        """获取某视频的中间结果目录路径。"""
        return self._cache_root / "intermediate" / stem

    def _progress_path(self, stem: str) -> Path:
        """获取某视频的进度文件路径。"""
        return self._cache_root / "progress" / f"{stem}.json"

    def _has_l1_intermediate(self, stem: str, l1_idx: int) -> bool:
        """检查某 L1 段的中间 JSON 是否存在。"""
        path = self._intermediate_dir(stem) / f"l1_{l1_idx}.json"
        return path.is_file()

    def _save_l1_intermediate(self, stem: str, l1_node: L1Node, l1_idx: int) -> None:
        """将单个 L1 段的中间结果保存到 JSON 文件。"""
        dir_path = self._intermediate_dir(stem)
        dir_path.mkdir(parents=True, exist_ok=True)
        out_path = dir_path / f"l1_{l1_idx}.json"
        save_l1_json(str(out_path), l1_node)

    def _load_l1_intermediate(self, stem: str, l1_idx: int) -> Optional[L1Node]:
        """从中间 JSON 加载单个 L1 段，若不存在则返回 None。"""
        path = self._intermediate_dir(stem) / f"l1_{l1_idx}.json"
        if not path.is_file():
            return None
        return load_l1_json(str(path))

    def _load_progress(self, stem: str) -> Optional[Dict[str, object]]:
        """加载某视频的进度文件（若不存在则返回 None）。"""
        path = self._progress_path(stem)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            try:
                data: Dict[str, object] = json.load(f)
            except json.JSONDecodeError:
                log_msg("WARNING", "进度文件 JSON 解析失败，忽略", path=str(path))
                return None
        return data

    def _save_progress(self, stem: str, total_l1: int, finished_l1_ids: set[int]) -> None:
        """将最新进度写回磁盘，支持断点续跑。"""
        path = self._progress_path(stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_id": stem,
            "total_l1": total_l1,
            "finished_l1_ids": sorted(finished_l1_ids),
            "updated_at": datetime.now().isoformat(),
        }
        if not path.is_file():
            payload["created_at"] = payload["updated_at"]
        else:
            # 尝试保留旧 created_at
            old = self._load_progress(stem)
            if old and isinstance(old.get("created_at"), str):
                payload["created_at"] = old["created_at"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log_msg(
            "INFO",
            "进度文件已更新",
            path=str(path),
            total_l1=total_l1,
            finished_l1=list(sorted(finished_l1_ids)),
        )

    def _cleanup_intermediate_and_progress(self, stem: str) -> None:
        """在最终 JSON 写入成功后清理中间结果与进度文件。"""
        # 清理 progress
        progress_path = self._progress_path(stem)
        if progress_path.is_file():
            try:
                progress_path.unlink()
            except OSError:
                log_msg("WARNING", "删除进度文件失败", path=str(progress_path))

        # 清理 intermediate 目录
        inter_dir = self._intermediate_dir(stem)
        if inter_dir.is_dir():
            for child in inter_dir.glob("l1_*.json"):
                try:
                    child.unlink()
                except OSError:
                    log_msg("WARNING", "删除 L1 中间 JSON 失败", path=str(child))
            try:
                # 目录可能仍有其它调试文件，忽略删除异常
                inter_dir.rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 内部方法：异步节点构建
    # ------------------------------------------------------------------

    async def _build_l2_video_async(
        self,
        video_path: str,
        clip_range: Tuple[float, float],
        l2_id: str,
        source_id: Optional[str],
        vlm_sem: asyncio.Semaphore,
    ) -> L2Node:
        """异步构建 L2 视频节点（代表帧 VLM 描述）。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            clip_range: L2 clip 时间区间 (start, end)，单位秒。
            l2_id: 节点 ID。
            source_id: 帧缓存目录名。
            vlm_sem: VLM 并发控制信号量。

        返回:
            L2Node（children 为空，由后续 L3 阶段填充）。

        实现细节:
            均匀采样 l2_representative_frames 帧，并行 ffmpeg 提取，
            async with vlm_sem 限制 VLM 并发量。
        """
        start_sec, end_sec = clip_range
        n_rep = self.config.l2_representative_frames

        if n_rep == 1:
            timestamps = [(start_sec + end_sec) / 2.0]
        else:
            step = (end_sec - start_sec) / (n_rep - 1)
            timestamps = [start_sec + i * step for i in range(n_rep)]

        video_stem = source_id if source_id is not None else self._source_stem(video_path)
        frame_dir = Path(self.config.cache_dir) / "frames" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()

        async def _extract_rep(ts: float) -> Optional[str]:
            frame_name = f"l2_{ts:.3f}.jpg"
            frame_path = str(frame_dir / frame_name)
            if os.path.isfile(frame_path):
                return frame_path
            success = await loop.run_in_executor(
                self._ffmpeg_pool,
                self._ffmpeg_extract_frame, video_path, ts, frame_path,
            )
            if not success:
                log_msg("WARNING", "L2 代表帧读取失败，跳过", timestamp=ts)
                return None
            return frame_path

        # 并发提取所有代表帧
        rep_results = await asyncio.gather(*[_extract_rep(ts) for ts in timestamps])
        rep_frames = [p for p in rep_results if p is not None]
        ensure(len(rep_frames) > 0, f"L2 节点 {l2_id} 代表帧提取结果为空")

        # VLM 调用受信号量保护
        async with vlm_sem:
            description = await self.vlm.chat_with_images_async(
                _L2_VIDEO_PROMPT, images=rep_frames
            )

        return L2Node(
            id=l2_id,
            description=description,
            embedding=None,
            time_range=clip_range,
        )

    async def _build_l3_task_async(
        self,
        video_path: str,
        l2_node: L2Node,
        clip_range: Tuple[float, float],
        source_id: str,
        l1_i: int,
        l2_j: int,
        vlm_sem: asyncio.Semaphore,
    ) -> L2Node:
        """异步 L3 任务：并发提帧 + 批次级并发 VLM 帧描述。

        参数:
            video_path: 视频文件路径或 CDN 流式 URL。
            l2_node: 已构建的 L2 节点。
            clip_range: L2 clip 时间区间。
            source_id: 帧缓存目录名。
            l1_i: 父 L1 索引。
            l2_j: 父 L2 索引。
            vlm_sem: VLM 并发控制信号量。

        返回:
            已填充 children 的 L2Node。

        实现细节:
            提帧阶段完全并行（受 ffmpeg 线程池限制）；
            VLM 调用阶段：12个批次同时提交（asyncio.gather），受信号量限流。
        """
        all_frames = await self._extract_frames_async(
            video_path, clip_range, self.config.l3_fps, source_id=source_id
        )
        l3_nodes = await self._build_l3_video_async(
            all_frames, l2_node.description, l1_i, l2_j, vlm_sem
        )
        l2_node.children = l3_nodes
        return l2_node

    async def _build_l3_video_async(
        self,
        frames: List[Tuple[str, float]],
        l2_description: str,
        l1_i: int,
        l2_j: int,
        vlm_sem: asyncio.Semaphore,
    ) -> List[L3Node]:
        """异步批次级并发构建 L3 节点（核心加速点）。

        参数:
            frames: [(frame_path, timestamp), ...]。
            l2_description: L2 节点描述，注入 prompt 上下文。
            l1_i: 父 L1 索引（用于节点 ID 生成）。
            l2_j: 父 L2 索引（用于节点 ID 生成）。
            vlm_sem: VLM 并发控制信号量。

        返回:
            L3Node 列表，每项对应一帧。

        实现细节:
            将全部帧按 _L3_BATCH_SIZE 分批，所有批次同时提交（asyncio.gather），
            每批通过 vlm_sem 限流，实现批次级真并发。
            对比旧版串行：12批 × 6s = 72s → 现在 ~6s（受信号量限流取最慢批次）。
        """
        ensure(len(frames) > 0, f"L3 帧列表为空 (l1={l1_i}, l2={l2_j})")

        # Phase 1: 构建所有批次的协程（同时提交，asyncio.gather 并发执行）
        batches: List[List[Tuple[str, float]]] = []
        for batch_start in range(0, len(frames), _L3_BATCH_SIZE):
            batches.append(frames[batch_start : batch_start + _L3_BATCH_SIZE])

        batch_results: List[List[str]] = list(
            await asyncio.gather(
                *[
                    self._call_vlm_batch_async(
                        batch, l2_description, l1_i, l2_j, vlm_sem
                    )
                    for batch in batches
                ]
            )
        )

        # Phase 2: 展平所有批次描述，构建 L3 节点
        all_descriptions: List[str] = [
            desc for batch_descs in batch_results for desc in batch_descs
        ]

        nodes: List[L3Node] = []
        for k, (desc, (frame_path, ts)) in enumerate(zip(all_descriptions, frames)):
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

    async def _call_vlm_batch_async(
        self,
        batch: List[Tuple[str, float]],
        l2_description: str,
        l1_i: int,
        l2_j: int,
        vlm_sem: asyncio.Semaphore,
    ) -> List[str]:
        """异步单批次 VLM 调用（≤ _L3_BATCH_SIZE 帧），解析失败时逐帧 fallback。

        参数:
            batch: 本批帧列表 [(frame_path, ts), ...]。
            l2_description: L2 描述，用于 prompt 和 fallback prompt。
            l1_i: 父 L1 索引（日志用）。
            l2_j: 父 L2 索引（日志用）。
            vlm_sem: VLM 并发控制信号量。

        返回:
            与 batch 等长的描述文本列表。

        实现细节:
            async with vlm_sem 确保并发量不超过 config.concurrency。
            fallback 时逐帧并发（asyncio.gather），同样受信号量保护。
        """
        batch_paths = [fp for fp, _ in batch]
        n = len(batch_paths)
        prompt = _L3_VIDEO_PROMPT.format(l2_description=l2_description, n=n)

        # Phase 1: 尝试批量调用
        try:
            async with vlm_sem:
                raw = await self.vlm.chat_with_images_async(prompt, images=batch_paths)
            descriptions = self._parse_json_descriptions(raw, n)
            if descriptions is not None:
                return descriptions
            log_msg(
                "WARNING",
                "L3 小批量 VLM JSON 解析失败，对本批逐帧 fallback",
                l1=l1_i,
                l2=l2_j,
                batch_n=n,
                raw_preview=raw[:100],
            )
        except Exception as exc:
            log_msg(
                "WARNING",
                f"L3 小批量 VLM 调用异常，对本批逐帧 fallback: {exc}",
                l1=l1_i,
                l2=l2_j,
                batch_n=n,
            )

        # Phase 2: 逐帧 fallback（并发，受信号量保护）
        single_prompt = _L3_SINGLE_PROMPT.format(l2_description=l2_description)

        async def _single_frame(fp: str) -> str:
            async with vlm_sem:
                return await self.vlm.chat_with_images_async(single_prompt, images=[fp])

        return list(await asyncio.gather(*[_single_frame(fp) for fp in batch_paths]))

    async def _build_l1_video_async(
        self,
        l2_children: List[L2Node],
        l1_id: str,
        l1_range: Tuple[float, float],
        vlm_sem: asyncio.Semaphore,
    ) -> L1Node:
        """异步构建 L1 节点（LLM 文本摘要）。

        参数:
            l2_children: 该 L1 节点下的所有 L2 节点。
            l1_id: 节点 ID。
            l1_range: L1 时间区间 (start, end)，单位秒。
            vlm_sem: VLM 并发控制信号量。

        返回:
            L1Node（children 已赋值）。
        """
        ensure(len(l2_children) > 0, f"L1 节点 {l1_id} 没有 L2 子节点")
        l2_texts = "\n".join(f"- {node.description}" for node in l2_children)
        prompt = _L1_VIDEO_PROMPT.format(l2_texts=l2_texts)

        async with vlm_sem:
            summary = await self.vlm.chat_async(prompt)

        log_msg("INFO", "L1 触发", l1_id=l1_id)
        return L1Node(
            id=l1_id,
            summary=summary,
            embedding=None,
            time_range=l1_range,
            children=l2_children,
        )

    # ------------------------------------------------------------------
    # 内部方法：JSON 解析（同步，纯 CPU）
    # ------------------------------------------------------------------

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

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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger_system import ensure, log_json, log_msg
from video_tree_trm.config import TreeConfig
from video_tree_trm.embeddings import EmbeddingModel
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
    转化为三层 TreeIndex。所有描述通过 VLM 生成，所有嵌入使用 text_embed。

    属性:
        embed: 嵌入模型（冻结）。
        vlm: VLM/LLM 客户端（用于图文和纯文本调用）。
        config: 树构建配置。
    """

    def __init__(
        self,
        embed_model: EmbeddingModel,
        vlm: LLMClient,
        config: TreeConfig,
    ) -> None:
        """初始化视频树构建器。

        参数:
            embed_model: 已初始化的嵌入模型（EmbeddingModel）。
            vlm: 已初始化的 VLM/LLM 客户端（LLMClient），需支持 chat_with_images。
            config: 树构建配置（TreeConfig），关键字段：
                    l1_segment_duration, l2_clip_duration, l3_fps,
                    l2_representative_frames, cache_dir。
        """
        self.embed = embed_model
        self.vlm = vlm
        self.config = config

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build(self, video_path: str) -> TreeIndex:
        """将长视频构建为三层 TreeIndex。

        参数:
            video_path: 输入视频文件路径（支持 .mp4/.avi/.mkv 等 OpenCV 可读格式）。

        返回:
            三层 TreeIndex 对象。

        实现细节:
            Step 1: _segment_video → List[Tuple[float, float]]（L1 时间区间）
            Step 2: L2 先行 — 每个 L2 clip 用 VLM 生成描述 + 嵌入
            Step 3: L3 向下 — 注入 L2 上下文，VLM 批量帧描述 + 嵌入
            Step 4: L1 向上 — 聚合 L2 描述，LLM 生成 L1 摘要 + 嵌入
            Step 5: 组装 TreeIndex 并写入日志
        """
        ensure(os.path.isfile(video_path), f"视频文件不存在: {video_path}")
        log_msg("INFO", "开始构建视频树索引", video_path=video_path)

        # Phase 1: 时间切分 → L1 区间列表
        l1_ranges = self._segment_video(video_path)
        ensure(len(l1_ranges) > 0, "视频时间切分结果为空")
        log_msg("INFO", "视频切分完成", l1_count=len(l1_ranges))

        l1_nodes: List[L1Node] = []

        for i, l1_range in enumerate(l1_ranges):
            l2_clips = self._get_l2_clips(l1_range)
            l2_nodes: List[L2Node] = []

            for j, clip_range in enumerate(l2_clips):
                l2_id = f"l1_{i}_l2_{j}"

                # Phase 2: L2 先行 — VLM 代表帧描述
                l2_node = self._build_l2_video(video_path, clip_range, l2_id)

                # Phase 3: L3 向下 — 提取所有帧，注入 L2 上下文
                all_frames = self._extract_frames(
                    video_path, clip_range, self.config.l3_fps
                )
                l3_nodes = self._build_l3_video(all_frames, l2_node.description, i, j)
                l2_node.children = l3_nodes
                l2_nodes.append(l2_node)
                log_msg(
                    "INFO",
                    "L2 节点构建完成",
                    l2_id=l2_id,
                    l3_count=len(l3_nodes),
                )

            # Phase 4: L1 向上 — 聚合 L2 描述
            l1_node = self._build_l1_video(l2_nodes, f"l1_{i}", l1_range)
            l1_nodes.append(l1_node)
            log_msg("INFO", "L1 节点构建完成", l1_id=f"l1_{i}", l2_count=len(l2_nodes))

        # Phase 5: 组装 TreeIndex
        metadata = IndexMeta(
            source_path=video_path,
            modality="video",
            embed_model=self.embed._model_name,
            embed_dim=self.embed.dim,
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
                "embed_dim": self.embed.dim,
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

    def _segment_video(self, video_path: str) -> List[Tuple[float, float]]:
        """读取视频总时长，按固定步长切分为 L1 时间区间列表。

        参数:
            video_path: 视频文件路径。

        返回:
            L1 时间区间列表，每项为 (start_sec, end_sec)。
            最后一段可能短于 l1_segment_duration。

        实现细节:
            使用 cv2.VideoCapture 读取总帧数和 FPS 计算总时长，
            按 config.l1_segment_duration 固定步长切分。
        """
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
    ) -> List[Tuple[str, float]]:
        """按指定 fps 提取时间范围内的帧，保存到 cache 目录。

        参数:
            video_path: 视频文件路径。
            time_range: 提取时间区间 (start_sec, end_sec)。
            fps: 提取帧率（帧/秒）。

        返回:
            [(frame_path, timestamp_sec), ...]，按时间顺序排列。
            已存在的帧文件直接复用，不重复提取。

        实现细节:
            帧保存路径: {cache_dir}/frames/{video_stem}/{start:.1f}_{ts:.3f}.jpg
            使用 cv2.VideoCapture.set(CAP_PROP_POS_MSEC) 精确定位帧位置。
        """
        video_stem = Path(video_path).stem
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
    ) -> L2Node:
        """构建 L2 视频节点（代表帧 VLM 描述 + 嵌入）。

        参数:
            video_path: 视频文件路径。
            clip_range: L2 clip 时间区间 (start, end)，单位秒。
            l2_id: 节点 ID。

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

        video_stem = Path(video_path).stem
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
        embedding = self.embed.embed(description)[0]  # [D]
        ensure(
            embedding.shape == (self.embed.dim,),
            f"L2 嵌入维度异常: {embedding.shape}，期望 ({self.embed.dim},)",
        )
        return L2Node(
            id=l2_id,
            description=description,
            embedding=embedding.astype(np.float32),
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

        # Phase 2: 批量嵌入所有帧描述
        embeddings = self.embed.embed(descriptions)  # [N, D]
        ensure(
            embeddings.shape == (n, self.embed.dim),
            f"L3 嵌入矩阵形状异常: {embeddings.shape}，期望 ({n}, {self.embed.dim})",
        )

        nodes: List[L3Node] = []
        for k, (desc, emb, (frame_path, ts)) in enumerate(
            zip(descriptions, embeddings, frames)
        ):
            nodes.append(
                L3Node(
                    id=f"l1_{l1_i}_l2_{l2_j}_l3_{k}",
                    description=desc,
                    embedding=emb.astype(np.float32),
                    raw_content=None,
                    frame_path=frame_path,
                    timestamp=ts,
                )
            )
        return nodes

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
        embedding = self.embed.embed(summary)[0]  # [D]
        ensure(
            embedding.shape == (self.embed.dim,),
            f"L1 嵌入维度异常: {embedding.shape}，期望 ({self.embed.dim},)",
        )
        return L1Node(
            id=l1_id,
            summary=summary,
            embedding=embedding.astype(np.float32),
            time_range=l1_range,
            children=l2_children,
        )

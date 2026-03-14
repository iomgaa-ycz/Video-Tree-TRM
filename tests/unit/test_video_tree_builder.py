"""
VideoTreeBuilder 单元测试
=========================
覆盖视频树构建的各个子方法和完整流程。

测试策略:
    - mock cv2.VideoCapture 避免依赖真实视频（_segment_video 等）
    - 使用 cv2 合成小视频进行帧提取和集成测试（tiny_video fixture）
    - mock VLM/embed 依赖，隔离外部 API
    - 测试 L3 批量降级路径（JSON 解析失败时退回逐帧调用）
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from video_tree_trm.config import TreeConfig
from video_tree_trm.tree_index import L1Node, L2Node, L3Node
from video_tree_trm.video_tree_builder import VideoTreeBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_config(tmp_path: Path) -> TreeConfig:
    """构建测试用 TreeConfig，cache_dir 指向临时目录。"""
    return TreeConfig(
        max_paragraphs_per_l2=5,
        l1_segment_duration=30.0,
        l2_clip_duration=10.0,
        l3_fps=1.0,
        l2_representative_frames=3,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def mock_embed() -> MagicMock:
    """返回 mock 嵌入模型，embed() 返回全 1 向量。"""
    embed = MagicMock()
    embed.dim = 4
    embed._model_name = "mock-embed"

    def _embed(texts):
        """根据输入类型返回 [1, D] 或 [N, D] 的 float32 数组。"""
        if isinstance(texts, str):
            return np.ones((1, 4), dtype=np.float32)
        n = len(texts)
        return np.ones((n, 4), dtype=np.float32)

    embed.embed.side_effect = _embed
    return embed


@pytest.fixture
def mock_vlm() -> MagicMock:
    """返回 mock VLM 客户端。"""
    vlm = MagicMock()
    vlm.chat.return_value = "这是一段精彩的视频内容摘要。"
    vlm.chat_with_images.return_value = "这帧画面中有人物在移动。"
    return vlm


@pytest.fixture
def builder(
    mock_embed: MagicMock, mock_vlm: MagicMock, tree_config: TreeConfig
) -> VideoTreeBuilder:
    """构建测试用 VideoTreeBuilder 实例。"""
    return VideoTreeBuilder(
        embed_model=mock_embed,
        vlm=mock_vlm,
        config=tree_config,
    )


@pytest.fixture
def tiny_video(tmp_path: Path) -> str:
    """用 cv2 生成 30 帧合成彩色视频（10fps，时长 3 秒），返回路径。

    视频规格:
        - 分辨率: 64×48
        - 帧率: 10fps
        - 时长: 3 秒（30 帧）
        - 内容: 随机彩色帧
    """
    video_path = str(tmp_path / "tiny.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 48))
    for _ in range(30):
        frame = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# 测试：_segment_video — 时间切分
# ---------------------------------------------------------------------------


def test_segment_video_fixed_step(builder: VideoTreeBuilder, tmp_path: Path) -> None:
    """mock cv2.VideoCapture（总时长=60s，l1_segment_duration=30s），
    验证切分出 2 个均等 L1 区间：(0,30),(30,60)。"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: (
        10.0 if prop == cv2.CAP_PROP_FPS else 600.0  # 600帧/10fps = 60s
    )

    with patch("video_tree_trm.video_tree_builder.cv2.VideoCapture", return_value=mock_cap):
        ranges = builder._segment_video("fake.mp4")

    assert len(ranges) == 2
    assert ranges[0] == (0.0, 30.0)
    assert ranges[1] == (30.0, 60.0)


def test_segment_video_uneven(builder: VideoTreeBuilder) -> None:
    """总时长不能被 l1_segment_duration 整除时，最后一段应短于步长。"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # 75s = 750帧 / 10fps
    mock_cap.get.side_effect = lambda prop: (
        10.0 if prop == cv2.CAP_PROP_FPS else 750.0
    )

    with patch("video_tree_trm.video_tree_builder.cv2.VideoCapture", return_value=mock_cap):
        ranges = builder._segment_video("fake.mp4")

    assert len(ranges) == 3
    assert ranges[0] == (0.0, 30.0)
    assert ranges[1] == (30.0, 60.0)
    assert abs(ranges[2][0] - 60.0) < 1e-6
    assert abs(ranges[2][1] - 75.0) < 1e-6


# ---------------------------------------------------------------------------
# 测试：_get_l2_clips — L2 切分
# ---------------------------------------------------------------------------


def test_get_l2_clips_even(builder: VideoTreeBuilder) -> None:
    """l1=(0,30)，l2_duration=10 → 3 clips 均等：(0,10),(10,20),(20,30)。"""
    clips = builder._get_l2_clips((0.0, 30.0))
    assert len(clips) == 3
    assert clips[0] == (0.0, 10.0)
    assert clips[1] == (10.0, 20.0)
    assert clips[2] == (20.0, 30.0)


def test_get_l2_clips_uneven(builder: VideoTreeBuilder) -> None:
    """l1=(0,25)，l2_duration=10 → 3 clips，最后一段为 5s。"""
    clips = builder._get_l2_clips((0.0, 25.0))
    assert len(clips) == 3
    assert clips[2] == (20.0, 25.0)


def test_get_l2_clips_shorter_than_step(builder: VideoTreeBuilder) -> None:
    """L1 区间短于 l2_clip_duration 时，返回 1 个 clip。"""
    clips = builder._get_l2_clips((0.0, 5.0))
    assert len(clips) == 1
    assert clips[0] == (0.0, 5.0)


# ---------------------------------------------------------------------------
# 测试：_extract_frames — 帧提取
# ---------------------------------------------------------------------------


def test_extract_frames_saves_files(
    builder: VideoTreeBuilder, tiny_video: str, tmp_path: Path
) -> None:
    """使用真实合成视频（3s），提取 1fps 帧，验证返回路径和文件存在。"""
    frames = builder._extract_frames(tiny_video, (0.0, 3.0), fps=1.0)

    assert len(frames) >= 1
    for frame_path, ts in frames:
        assert os.path.isfile(frame_path), f"帧文件不存在: {frame_path}"
        assert ts >= 0.0


def test_extract_frames_cache_reuse(
    builder: VideoTreeBuilder, tiny_video: str
) -> None:
    """第二次提取同一区间时，帧文件应直接复用（不重复写磁盘）。"""
    frames1 = builder._extract_frames(tiny_video, (0.0, 2.0), fps=1.0)
    assert len(frames1) >= 1

    # 记录文件修改时间
    mtimes_before = [os.path.getmtime(fp) for fp, _ in frames1]

    frames2 = builder._extract_frames(tiny_video, (0.0, 2.0), fps=1.0)
    mtimes_after = [os.path.getmtime(fp) for fp, _ in frames2]

    assert frames1 == frames2
    assert mtimes_before == mtimes_after, "缓存帧文件被重复写入"


def test_extract_frames_empty_range(
    builder: VideoTreeBuilder, tiny_video: str
) -> None:
    """时间范围内无有效时间戳时，返回空列表。"""
    # 起始=结束，无时间戳
    frames = builder._extract_frames(tiny_video, (1.0, 1.0), fps=1.0)
    assert frames == []


# ---------------------------------------------------------------------------
# 测试：_build_l2_video — L2 节点构建
# ---------------------------------------------------------------------------


def test_build_l2_video_node_structure(
    builder: VideoTreeBuilder, tiny_video: str, mock_vlm: MagicMock
) -> None:
    """验证 L2Node 字段：description 非空、embedding shape 正确、time_range 正确。"""
    mock_vlm.chat_with_images.return_value = "片段展示了室内场景的变化。"
    l2_node = builder._build_l2_video(tiny_video, (0.0, 2.0), "l1_0_l2_0")

    assert isinstance(l2_node, L2Node)
    assert l2_node.id == "l1_0_l2_0"
    assert len(l2_node.description) > 0
    assert l2_node.embedding.shape == (4,)
    assert l2_node.embedding.dtype == np.float32
    assert l2_node.time_range == (0.0, 2.0)
    assert l2_node.children == []  # 调用方填充


def test_build_l2_video_representative_frames_count(
    builder: VideoTreeBuilder, tiny_video: str, mock_vlm: MagicMock
) -> None:
    """验证 VLM 被调用时传入的图像数不超过 l2_representative_frames。"""
    mock_vlm.chat_with_images.return_value = "描述内容。"
    builder._build_l2_video(tiny_video, (0.0, 3.0), "l1_0_l2_0")

    call_args = mock_vlm.chat_with_images.call_args
    images_passed = call_args.kwargs.get("images", call_args.args[1] if len(call_args.args) > 1 else [])
    assert len(images_passed) <= builder.config.l2_representative_frames


# ---------------------------------------------------------------------------
# 测试：_build_l3_video — L3 节点构建
# ---------------------------------------------------------------------------


def _make_frames(n: int, tmp_path: Path) -> List[tuple]:
    """创建 n 个临时 JPEG 帧文件，返回 [(path, ts), ...]。"""
    frames = []
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(exist_ok=True)
    for i in range(n):
        frame_path = str(frame_dir / f"frame_{i}.jpg")
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.imwrite(frame_path, img)
        frames.append((frame_path, float(i)))
    return frames


def test_build_l3_video_batch_success(
    builder: VideoTreeBuilder, mock_vlm: MagicMock, tmp_path: Path
) -> None:
    """mock VLM 返回合法 JSON 数组，验证 L3Node 列表结构正确。"""
    frames = _make_frames(2, tmp_path)
    mock_vlm.chat_with_images.return_value = json.dumps(["帧1描述内容", "帧2描述内容"])

    nodes = builder._build_l3_video(frames, "片段整体描述", l1_i=0, l2_j=0)

    assert len(nodes) == 2
    for k, node in enumerate(nodes):
        assert isinstance(node, L3Node)
        assert node.id == f"l1_0_l2_0_l3_{k}"
        assert len(node.description) > 0
        assert node.embedding.shape == (4,)
        assert node.embedding.dtype == np.float32
        assert node.frame_path == frames[k][0]
        assert node.timestamp == float(k)
        assert node.raw_content is None


def test_build_l3_video_batch_fallback(
    builder: VideoTreeBuilder, mock_vlm: MagicMock, tmp_path: Path
) -> None:
    """mock VLM 第一次返回非 JSON 字符串，验证降级逐帧调用（call_count == n+1）。

    第一次 = 批量调用（失败），后 n 次 = 逐帧调用。
    """
    n = 3
    frames = _make_frames(n, tmp_path)
    # 第一次返回无效 JSON，后续逐帧返回正常描述
    mock_vlm.chat_with_images.side_effect = (
        ["这不是一个JSON数组，无法解析"] + [f"第{i}帧描述" for i in range(n)]
    )

    nodes = builder._build_l3_video(frames, "片段整体描述", l1_i=0, l2_j=0)

    assert len(nodes) == n
    # 1次批量 + n次逐帧
    assert mock_vlm.chat_with_images.call_count == n + 1
    for node in nodes:
        assert len(node.description) > 0


def test_build_l3_video_json_length_mismatch_fallback(
    builder: VideoTreeBuilder, mock_vlm: MagicMock, tmp_path: Path
) -> None:
    """VLM 返回 JSON 但长度不匹配时，也应降级逐帧调用。"""
    n = 3
    frames = _make_frames(n, tmp_path)
    # 只返回 2 项，但期望 3 项
    mock_vlm.chat_with_images.side_effect = (
        [json.dumps(["描述1", "描述2"])] + [f"帧{i}" for i in range(n)]
    )

    nodes = builder._build_l3_video(frames, "片段描述", l1_i=0, l2_j=0)

    assert len(nodes) == n
    assert mock_vlm.chat_with_images.call_count == n + 1


# ---------------------------------------------------------------------------
# 测试：_build_l1_video — L1 节点构建
# ---------------------------------------------------------------------------


def test_build_l1_video_node_structure(
    builder: VideoTreeBuilder, mock_vlm: MagicMock, mock_embed: MagicMock
) -> None:
    """验证 L1Node 字段：summary 非空、time_range 正确、children 已赋值。"""
    mock_vlm.chat.return_value = "这段视频涵盖了户外活动和室内场景的切换。"

    l2_children = [
        L2Node(
            id=f"l1_0_l2_{j}",
            description=f"L2描述{j}",
            embedding=np.ones(4, dtype=np.float32),
            time_range=(j * 10.0, (j + 1) * 10.0),
        )
        for j in range(3)
    ]

    l1_node = builder._build_l1_video(l2_children, "l1_0", (0.0, 30.0))

    assert isinstance(l1_node, L1Node)
    assert l1_node.id == "l1_0"
    assert len(l1_node.summary) > 0
    assert l1_node.time_range == (0.0, 30.0)
    assert l1_node.children is l2_children
    assert l1_node.embedding.shape == (4,)
    assert l1_node.embedding.dtype == np.float32


def test_build_l1_video_prompt_contains_l2_descriptions(
    builder: VideoTreeBuilder, mock_vlm: MagicMock
) -> None:
    """验证 L1 摘要的 prompt 包含所有 L2 描述文本。"""
    mock_vlm.chat.return_value = "综合摘要内容。"
    l2_descriptions = ["片段A描述", "片段B描述", "片段C描述"]
    l2_children = [
        L2Node(
            id=f"l1_0_l2_{j}",
            description=desc,
            embedding=np.ones(4, dtype=np.float32),
            time_range=(j * 10.0, (j + 1) * 10.0),
        )
        for j, desc in enumerate(l2_descriptions)
    ]

    builder._build_l1_video(l2_children, "l1_0", (0.0, 30.0))

    call_prompt = mock_vlm.chat.call_args.args[0]
    for desc in l2_descriptions:
        assert desc in call_prompt, f"L2 描述 '{desc}' 未出现在 L1 prompt 中"


# ---------------------------------------------------------------------------
# 测试：build 完整流程（集成测试，mock VLM）
# ---------------------------------------------------------------------------


def test_build_full_integration(
    builder: VideoTreeBuilder,
    tiny_video: str,
    mock_vlm: MagicMock,
    mock_embed: MagicMock,
    tmp_path: Path,
) -> None:
    """用合成视频（3s）+ mock VLM 验证完整 TreeIndex 三层结构。

    配置：l1_segment_duration=2s，l2_clip_duration=1s
    预期：至少 1 个 L1，每 L1 至少 1 个 L2，每 L2 至少 1 个 L3。
    """
    # 调整 config 使 3s 视频能切出多个节点
    builder.config.l1_segment_duration = 2.0
    builder.config.l2_clip_duration = 1.0

    # VLM 批量调用返回 JSON 数组（按帧数动态生成）
    def vlm_side_effect(prompt, images=None):
        if images and len(images) > 1:
            # L3 批量调用：返回 JSON
            return json.dumps([f"帧{i}描述" for i in range(len(images))])
        return "这是 VLM 的描述文本。"

    mock_vlm.chat_with_images.side_effect = vlm_side_effect
    mock_vlm.chat.return_value = "L1 整体摘要内容。"

    index = builder.build(tiny_video)

    # 验证元数据
    assert index.metadata.modality == "video"
    assert index.metadata.source_path == tiny_video
    assert index.metadata.embed_dim == 4

    # 验证三层结构非空
    assert len(index.roots) >= 1, "应有至少 1 个 L1 节点"
    for l1 in index.roots:
        assert len(l1.children) >= 1, f"L1 {l1.id} 应有至少 1 个 L2 子节点"
        assert l1.time_range is not None
        for l2 in l1.children:
            assert len(l2.children) >= 1, f"L2 {l2.id} 应有至少 1 个 L3 子节点"
            assert l2.time_range is not None
            for l3 in l2.children:
                assert l3.frame_path is not None
                assert l3.timestamp is not None
                assert l3.embedding.shape == (4,)


def test_build_saves_output_md(
    builder: VideoTreeBuilder,
    tiny_video: str,
    mock_vlm: MagicMock,
    tmp_path: Path,
) -> None:
    """构建完成后，将执行摘要保存为 Markdown（CLAUDE.md 规范）。"""
    builder.config.l1_segment_duration = 2.0
    builder.config.l2_clip_duration = 1.0

    def vlm_side_effect(prompt, images=None):
        if images and len(images) > 1:
            return json.dumps([f"帧{i}描述" for i in range(len(images))])
        return "VLM 描述内容。"

    mock_vlm.chat_with_images.side_effect = vlm_side_effect
    mock_vlm.chat.return_value = "L1 摘要。"

    index = builder.build(tiny_video)

    # 保存 Markdown 输出
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "video_tree_builder"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"build_video_{ts}.md"

    total_l2 = sum(len(r.children) for r in index.roots)
    total_l3 = sum(len(l2.children) for r in index.roots for l2 in r.children)

    lines = [
        f"# Agent 测试: test_build_saves_output_md",
        f"## 任务: VideoTreeBuilder.build() 完整流程验证",
        f"",
        f"## 输入",
        f"- 视频路径: `{tiny_video}`",
        f"- l1_segment_duration: {builder.config.l1_segment_duration}s",
        f"- l2_clip_duration: {builder.config.l2_clip_duration}s",
        f"- l3_fps: {builder.config.l3_fps}",
        f"",
        f"## 输出结构",
        f"- L1 节点数: {len(index.roots)}",
        f"- L2 节点数: {total_l2}",
        f"- L3 节点数: {total_l3}",
        f"- embed_dim: {index.metadata.embed_dim}",
        f"",
        f"## L1 详情",
    ]
    for l1 in index.roots:
        lines.append(f"### {l1.id} (time_range={l1.time_range})")
        lines.append(f"- summary: {l1.summary[:80]}...")
        for l2 in l1.children:
            lines.append(
                f"  - {l2.id} [{l2.time_range}]: {l2.description[:60]}... "
                f"({len(l2.children)} L3)"
            )
    lines += [
        "",
        "## 最终结果",
        "✅ TreeIndex 构建成功，三层结构完整。",
        f"",
        f"输出文件: `{output_path}`",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[测试输出] {output_path}")
    assert output_path.is_file()

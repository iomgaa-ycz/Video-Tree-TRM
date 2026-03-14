"""
test_answer_generator.py — AnswerGenerator / token_f1 单元测试
==============================================================
使用 unittest.mock.MagicMock 模拟 LLMClient，不调用真实 API。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from video_tree_trm.answer_generator import AnswerGenerator, token_f1
from video_tree_trm.tree_index import IndexMeta, L1Node, L2Node, L3Node, TreeIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 4  # 最小嵌入维度，仅用于构造 TreeIndex


def _meta(modality: str) -> IndexMeta:
    return IndexMeta(
        source_path="dummy",
        modality=modality,
        embed_model="test",
        embed_dim=D,
    )


def _l3(
    idx: int,
    raw_content: str | None = None,
    frame_path: str | None = None,
    description: str = "",
) -> L3Node:
    return L3Node(
        id=f"l3_{idx}",
        description=description,
        embedding=np.zeros(D, dtype=np.float32),
        raw_content=raw_content,
        frame_path=frame_path,
    )


def _build_tree(modality: str, l3_nodes: list[L3Node]) -> TreeIndex:
    """构造最小 TreeIndex：1 L1 × 1 L2 × n L3。"""
    l2 = L2Node(
        id="l2_0",
        description="L2",
        embedding=np.zeros(D, dtype=np.float32),
        children=l3_nodes,
    )
    l1 = L1Node(
        id="l1_0",
        summary="L1",
        embedding=np.zeros(D, dtype=np.float32),
        children=[l2],
    )
    return TreeIndex(metadata=_meta(modality), roots=[l1])


@pytest.fixture
def small_tree_text() -> TreeIndex:
    """文本模式：两个 L3 节点，含 raw_content。"""
    return _build_tree(
        "text",
        [
            _l3(0, raw_content="段落一内容"),
            _l3(1, raw_content="段落二内容"),
        ],
    )


@pytest.fixture
def small_tree_video() -> TreeIndex:
    """视频模式：两个 L3 节点，含 frame_path + description。"""
    return _build_tree(
        "video",
        [
            _l3(0, frame_path="frame_0.jpg", description="第一帧描述"),
            _l3(1, frame_path="frame_1.jpg", description="第二帧描述"),
        ],
    )


def _mock_llm(return_value: str = "模拟答案") -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = return_value
    return llm


def _mock_vlm(return_value: str = "视觉答案") -> MagicMock:
    vlm = MagicMock()
    vlm.chat_with_images.return_value = return_value
    return vlm


# ---------------------------------------------------------------------------
# AnswerGenerator — 文本模式测试
# ---------------------------------------------------------------------------


def test_generate_text_mode(small_tree_text: TreeIndex) -> None:
    """文本模式：llm.chat 被调用，prompt 包含 raw_content。"""
    llm = _mock_llm("答案A")
    gen = AnswerGenerator(llm=llm)
    answer = gen.generate("问题", paths=[(0, 0, 0), (0, 0, 1)], tree=small_tree_text)

    assert answer == "答案A"
    llm.chat.assert_called_once()
    prompt = llm.chat.call_args[0][0]
    assert "段落一内容" in prompt
    assert "段落二内容" in prompt
    assert "问题" in prompt


def test_generate_text_mode_skips_no_content(small_tree_text: TreeIndex) -> None:
    """raw_content=None 的节点应被跳过，不出现在 prompt 中。"""
    # 修改第 2 个节点的 raw_content 为 None
    small_tree_text.roots[0].children[0].children[1].raw_content = None

    llm = _mock_llm()
    gen = AnswerGenerator(llm=llm)
    gen.generate("问题", paths=[(0, 0, 0), (0, 0, 1)], tree=small_tree_text)

    prompt = llm.chat.call_args[0][0]
    assert "段落一内容" in prompt
    assert "段落二内容" not in prompt


# ---------------------------------------------------------------------------
# AnswerGenerator — 视频模式测试
# ---------------------------------------------------------------------------


def test_generate_video_mode(small_tree_video: TreeIndex) -> None:
    """视频模式：vlm.chat_with_images 被调用，含 frames + captions。"""
    llm = _mock_llm()
    vlm = _mock_vlm("视觉答案")
    gen = AnswerGenerator(llm=llm, vlm=vlm)

    answer = gen.generate(
        "视频问题", paths=[(0, 0, 0), (0, 0, 1)], tree=small_tree_video
    )

    assert answer == "视觉答案"
    vlm.chat_with_images.assert_called_once()

    call_kwargs = vlm.chat_with_images.call_args
    prompt = call_kwargs[0][0]
    images = (
        call_kwargs[1]["images"] if "images" in call_kwargs[1] else call_kwargs[0][1]
    )

    assert "第一帧描述" in prompt
    assert "第二帧描述" in prompt
    assert "frame_0.jpg" in images
    assert "frame_1.jpg" in images


def test_generate_video_mode_skips_no_frame(small_tree_video: TreeIndex) -> None:
    """frame_path=None 的节点不进入 images 列表。"""
    small_tree_video.roots[0].children[0].children[1].frame_path = None

    llm = _mock_llm()
    vlm = _mock_vlm()
    gen = AnswerGenerator(llm=llm, vlm=vlm)
    gen.generate("问题", paths=[(0, 0, 0), (0, 0, 1)], tree=small_tree_video)

    call_args = vlm.chat_with_images.call_args
    images = call_args[1]["images"] if "images" in call_args[1] else call_args[0][1]
    assert "frame_0.jpg" in images
    assert "frame_1.jpg" not in images


def test_generate_video_fallback_no_frames() -> None:
    """所有节点 frame_path=None 时，退化为 llm.chat（不调用 vlm）。"""
    tree = _build_tree(
        "video",
        [_l3(0, frame_path=None, description="纯描述，无帧")],
    )
    llm = _mock_llm("退化答案")
    vlm = _mock_vlm()
    gen = AnswerGenerator(llm=llm, vlm=vlm)

    answer = gen.generate("问题", paths=[(0, 0, 0)], tree=tree)

    assert answer == "退化答案"
    llm.chat.assert_called_once()
    vlm.chat_with_images.assert_not_called()
    prompt = llm.chat.call_args[0][0]
    assert "纯描述，无帧" in prompt


def test_generate_video_requires_vlm() -> None:
    """视频模式且 vlm=None 且存在帧路径时，应抛出 ValueError。"""
    tree = _build_tree(
        "video",
        [_l3(0, frame_path="frame.jpg", description="帧描述")],
    )
    gen = AnswerGenerator(llm=_mock_llm(), vlm=None)

    with pytest.raises(ValueError):
        gen.generate("问题", paths=[(0, 0, 0)], tree=tree)


# ---------------------------------------------------------------------------
# token_f1 测试
# ---------------------------------------------------------------------------


def test_token_f1_exact_match() -> None:
    """完全相同字符串 → F1 = 1.0。"""
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_token_f1_no_overlap() -> None:
    """完全不同字符串 → F1 = 0.0。"""
    assert token_f1("cat", "dog") == pytest.approx(0.0)


def test_token_f1_partial() -> None:
    """部分重叠：验证 F1 计算正确。"""
    # pred = ["the", "cat"], gt = ["the", "dog"]
    # common = {"the": 1}，P = 1/2, R = 1/2, F1 = 0.5
    assert token_f1("the cat", "the dog") == pytest.approx(0.5)


def test_token_f1_empty() -> None:
    """prediction 或 ground_truth 为空 → 0.0。"""
    assert token_f1("", "reference") == pytest.approx(0.0)
    assert token_f1("prediction", "") == pytest.approx(0.0)
    assert token_f1("", "") == pytest.approx(0.0)

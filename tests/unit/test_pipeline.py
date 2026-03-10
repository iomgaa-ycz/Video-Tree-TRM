"""
test_pipeline.py — Pipeline 单元测试
======================================
使用 unittest.mock.MagicMock + patch 隔离所有外部依赖（无真实 API / 文件 IO）。
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from video_tree_trm.pipeline import Pipeline
from video_tree_trm.tree_index import IndexMeta, L1Node, L2Node, L3Node, TreeIndex


# ---------------------------------------------------------------------------
# 辅助：构造最小 Config Mock
# ---------------------------------------------------------------------------

D = 8  # 嵌入维度（与 RetrieverConfig 一致）


def _make_config(checkpoint: str | None = None) -> MagicMock:
    """返回一个 Mock Config，字段值与实际 dataclass 对齐。"""
    cfg = MagicMock()
    cfg.embed.model_name = "test-embed"
    cfg.embed.embed_dim = D
    cfg.retriever.checkpoint = checkpoint
    cfg.retriever.embed_dim = D
    cfg.retriever.num_heads = 2
    cfg.retriever.L_layers = 2
    cfg.retriever.L_cycles = 2
    cfg.retriever.max_rounds = 2
    cfg.retriever.ffn_expansion = 2.0
    cfg.tree.cache_dir = "/tmp/test_pipeline_cache"
    return cfg


def _make_small_tree() -> TreeIndex:
    """构造最小 1×1×1 TreeIndex，用于 query() 测试。"""
    meta = IndexMeta(
        source_path="dummy",
        modality="text",
        embed_model="test",
        embed_dim=D,
    )
    l3 = L3Node(
        id="l3_0",
        description="节点描述",
        embedding=np.zeros(D, dtype=np.float32),
        raw_content="节点内容",
    )
    l2 = L2Node(
        id="l2_0",
        description="L2",
        embedding=np.zeros(D, dtype=np.float32),
        children=[l3],
    )
    l1 = L1Node(
        id="l1_0",
        summary="L1",
        embedding=np.zeros(D, dtype=np.float32),
        children=[l2],
    )
    return TreeIndex(metadata=meta, roots=[l1])


# ---------------------------------------------------------------------------
# Patch 工厂：将所有子模块构造函数替换为 MagicMock
# ---------------------------------------------------------------------------

_PATCHES = [
    "video_tree_trm.pipeline.EmbeddingModel",
    "video_tree_trm.pipeline.LLMClient",
    "video_tree_trm.pipeline.RecursiveRetriever",
    "video_tree_trm.pipeline.AnswerGenerator",
    "video_tree_trm.pipeline.TextTreeBuilder",
    "video_tree_trm.pipeline.VideoTreeBuilder",
]


# ---------------------------------------------------------------------------
# Pipeline.__init__ 测试
# ---------------------------------------------------------------------------


def test_pipeline_init_components() -> None:
    """__init__ 后各属性（embed_model/llm/vlm/retriever/generator）均存在。"""
    cfg = _make_config()
    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(),
        AnswerGenerator=MagicMock(),
    ):
        p = Pipeline(cfg)

    assert hasattr(p, "embed_model"), "缺少 embed_model 属性"
    assert hasattr(p, "llm"), "缺少 llm 属性"
    assert hasattr(p, "vlm"), "缺少 vlm 属性"
    assert hasattr(p, "retriever"), "缺少 retriever 属性"
    assert hasattr(p, "generator"), "缺少 generator 属性"


def test_pipeline_init_no_checkpoint() -> None:
    """checkpoint=None 时 load_state_dict 不被调用。"""
    cfg = _make_config(checkpoint=None)
    mock_retriever_instance = MagicMock()
    MockRetriever = MagicMock(return_value=mock_retriever_instance)

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MockRetriever,
        AnswerGenerator=MagicMock(),
    ):
        Pipeline(cfg)

    mock_retriever_instance.load_state_dict.assert_not_called()


def test_pipeline_init_with_checkpoint(tmp_path: Path) -> None:
    """checkpoint 非 None 时 load_state_dict 被调用一次。"""
    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"")  # 创建空文件，使 os.path.isfile 返回 True

    cfg = _make_config(checkpoint=str(ckpt_file))
    mock_retriever_instance = MagicMock()
    MockRetriever = MagicMock(return_value=mock_retriever_instance)

    fake_state_dict = {"weight": torch.zeros(1)}

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MockRetriever,
        AnswerGenerator=MagicMock(),
    ), patch("video_tree_trm.pipeline.torch.load", return_value=fake_state_dict):
        Pipeline(cfg)

    mock_retriever_instance.load_state_dict.assert_called_once_with(fake_state_dict)


# ---------------------------------------------------------------------------
# Pipeline.build_index 测试
# ---------------------------------------------------------------------------


def test_build_index_text_calls_builder(tmp_path: Path) -> None:
    """文本模式调用 TextTreeBuilder.build，参数含文件内容。"""
    src = tmp_path / "doc.txt"
    src.write_text("文档内容", encoding="utf-8")

    cfg = _make_config()
    cfg.tree.cache_dir = str(tmp_path / "cache")

    mock_tree = MagicMock(spec=TreeIndex)
    mock_builder_instance = MagicMock()
    mock_builder_instance.build.return_value = mock_tree
    MockTextBuilder = MagicMock(return_value=mock_builder_instance)

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(),
        AnswerGenerator=MagicMock(),
        TextTreeBuilder=MockTextBuilder,
    ):
        p = Pipeline(cfg)
        result = p.build_index(str(src), modality="text")

    mock_builder_instance.build.assert_called_once()
    call_args = mock_builder_instance.build.call_args
    assert "文档内容" in call_args[0][0], "TextTreeBuilder.build 应传入文件内容"
    assert result is mock_tree


def test_build_index_video_calls_builder(tmp_path: Path) -> None:
    """视频模式调用 VideoTreeBuilder.build，参数为 source_path。"""
    cfg = _make_config()
    cfg.tree.cache_dir = str(tmp_path / "cache")

    mock_tree = MagicMock(spec=TreeIndex)
    mock_builder_instance = MagicMock()
    mock_builder_instance.build.return_value = mock_tree
    MockVideoBuilder = MagicMock(return_value=mock_builder_instance)

    video_path = "/fake/video.mp4"

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(),
        AnswerGenerator=MagicMock(),
        VideoTreeBuilder=MockVideoBuilder,
    ):
        p = Pipeline(cfg)
        result = p.build_index(video_path, modality="video")

    mock_builder_instance.build.assert_called_once_with(video_path)
    assert result is mock_tree


def test_build_index_cache_hit(tmp_path: Path) -> None:
    """缓存文件存在时直接 TreeIndex.load，不重新构建。"""
    cfg = _make_config()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cfg.tree.cache_dir = str(cache_dir)

    # 手动创建缓存文件（空文件即可让 isfile 返回 True）
    cache_file = cache_dir / "doc_text.pkl"
    cache_file.write_bytes(b"")

    mock_tree = MagicMock(spec=TreeIndex)
    mock_text_builder = MagicMock()

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(),
        AnswerGenerator=MagicMock(),
        TextTreeBuilder=mock_text_builder,
    ), patch("video_tree_trm.pipeline.TreeIndex.load", return_value=mock_tree) as mock_load:
        p = Pipeline(cfg)
        result = p.build_index(str(tmp_path / "doc.txt"), modality="text")

    mock_load.assert_called_once_with(str(cache_file))
    mock_text_builder.return_value.build.assert_not_called()
    assert result is mock_tree


def test_build_index_saves_cache(tmp_path: Path) -> None:
    """缓存不存在时构建后调用 tree.save。"""
    cfg = _make_config()
    cfg.tree.cache_dir = str(tmp_path / "cache")

    src = tmp_path / "doc.txt"
    src.write_text("内容", encoding="utf-8")

    mock_tree = MagicMock(spec=TreeIndex)
    mock_builder_instance = MagicMock()
    mock_builder_instance.build.return_value = mock_tree

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(),
        AnswerGenerator=MagicMock(),
        TextTreeBuilder=MagicMock(return_value=mock_builder_instance),
    ):
        p = Pipeline(cfg)
        p.build_index(str(src), modality="text")

    mock_tree.save.assert_called_once()
    saved_path: str = mock_tree.save.call_args[0][0]
    assert "doc_text.pkl" in saved_path, f"保存路径应含 'doc_text.pkl'，实际={saved_path}"


# ---------------------------------------------------------------------------
# Pipeline.query 测试
# ---------------------------------------------------------------------------


def test_query_embeds_question() -> None:
    """query() 调用 embed_model.embed_tensor(question)。"""
    cfg = _make_config()
    tree = _make_small_tree()

    mock_embed = MagicMock()
    mock_embed.embed_tensor.return_value = torch.zeros(1, D)
    MockEmbed = MagicMock(return_value=mock_embed)

    mock_retriever_instance = MagicMock()
    mock_retriever_instance.return_value = {"paths": [(0, 0, 0)], "num_rounds": 1}

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MockEmbed,
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(return_value=mock_retriever_instance),
        AnswerGenerator=MagicMock(),
    ):
        p = Pipeline(cfg)
        p.query("测试问题", tree)

    mock_embed.embed_tensor.assert_called_once_with("测试问题")


def test_query_calls_retriever() -> None:
    """query() 调用 retriever(q, tree)。"""
    cfg = _make_config()
    tree = _make_small_tree()

    q_tensor = torch.zeros(1, D)
    mock_embed = MagicMock()
    mock_embed.embed_tensor.return_value = q_tensor

    mock_retriever_instance = MagicMock()
    mock_retriever_instance.return_value = {"paths": [(0, 0, 0)], "num_rounds": 1}

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(return_value=mock_embed),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(return_value=mock_retriever_instance),
        AnswerGenerator=MagicMock(),
    ):
        p = Pipeline(cfg)
        p.query("测试问题", tree)

    mock_retriever_instance.assert_called_once()
    call_args = mock_retriever_instance.call_args
    # 第一个位置参数应为嵌入 Tensor，第二个为 tree
    assert call_args[0][1] is tree, "retriever 第二个参数应为 tree"


def test_query_returns_answer() -> None:
    """query() 返回 generator.generate 的返回值。"""
    cfg = _make_config()
    tree = _make_small_tree()

    mock_embed = MagicMock()
    mock_embed.embed_tensor.return_value = torch.zeros(1, D)

    mock_retriever_instance = MagicMock()
    mock_retriever_instance.return_value = {"paths": [(0, 0, 0)], "num_rounds": 1}

    mock_generator_instance = MagicMock()
    mock_generator_instance.generate.return_value = "生成的答案"

    with patch.multiple(
        "video_tree_trm.pipeline",
        EmbeddingModel=MagicMock(return_value=mock_embed),
        LLMClient=MagicMock(),
        RecursiveRetriever=MagicMock(return_value=mock_retriever_instance),
        AnswerGenerator=MagicMock(return_value=mock_generator_instance),
    ):
        p = Pipeline(cfg)
        answer = p.query("问题", tree)

    assert answer == "生成的答案", f"query() 应返回 generator 的结果，实际='{answer}'"
    mock_generator_instance.generate.assert_called_once_with(
        "问题", [(0, 0, 0)], tree
    )

"""
tree_index 单元测试
===================
覆盖: 嵌入矩阵提取、节点访问、边界检查、序列化往返、空树处理。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from video_tree_trm.tree_index import (
    IndexMeta,
    L1Node,
    L2Node,
    L3Node,
    TreeIndex,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 64


def _make_embed(seed: int = 0) -> np.ndarray:
    """生成固定种子的随机嵌入向量 [D]。"""
    rng = np.random.RandomState(seed)
    return rng.randn(EMBED_DIM).astype(np.float32)


def _make_meta() -> IndexMeta:
    return IndexMeta(
        source_path="test.mp4",
        modality="video",
        embed_model="test-model",
        embed_dim=EMBED_DIM,
    )


def _make_tree() -> TreeIndex:
    """构建一棵 2 x 2 x 3 的测试树。"""
    meta = _make_meta()
    roots = []
    seed = 0
    for i in range(2):
        l2_nodes = []
        for j in range(2):
            l3_nodes = [
                L3Node(
                    id=f"l3_{i}_{j}_{k}",
                    description=f"帧描述 {i}-{j}-{k}",
                    embedding=_make_embed(seed := seed + 1),
                )
                for k in range(3)
            ]
            l2_nodes.append(
                L2Node(
                    id=f"l2_{i}_{j}",
                    description=f"片段描述 {i}-{j}",
                    embedding=_make_embed(seed := seed + 1),
                    children=l3_nodes,
                )
            )
        roots.append(
            L1Node(
                id=f"l1_{i}",
                summary=f"摘要 {i}",
                embedding=_make_embed(seed := seed + 1),
                children=l2_nodes,
            )
        )
    return TreeIndex(metadata=meta, roots=roots)


# ---------------------------------------------------------------------------
# 测试: 嵌入矩阵提取
# ---------------------------------------------------------------------------


class TestEmbeddings:
    """嵌入矩阵提取方法测试。"""

    def test_l1_embeddings_shape(self) -> None:
        """l1_embeddings() 返回 [N1, D]。"""
        tree = _make_tree()
        emb = tree.l1_embeddings()
        assert emb.shape == (2, EMBED_DIM)
        assert emb.dtype == np.float32

    def test_l2_embeddings_of_shape(self) -> None:
        """l2_embeddings_of(idx) 返回 [N2, D]。"""
        tree = _make_tree()
        emb = tree.l2_embeddings_of(0)
        assert emb.shape == (2, EMBED_DIM)
        assert emb.dtype == np.float32

    def test_l3_embeddings_of_shape(self) -> None:
        """l3_embeddings_of(l1, l2) 返回 [N3, D]。"""
        tree = _make_tree()
        emb = tree.l3_embeddings_of(0, 1)
        assert emb.shape == (3, EMBED_DIM)
        assert emb.dtype == np.float32


# ---------------------------------------------------------------------------
# 测试: 节点访问
# ---------------------------------------------------------------------------


class TestGetNode:
    """节点访问方法测试。"""

    def test_get_node(self) -> None:
        """正确返回目标 L3Node。"""
        tree = _make_tree()
        node = tree.get_node(1, 0, 2)
        assert isinstance(node, L3Node)
        assert node.id == "l3_1_0_2"
        assert node.description == "帧描述 1-0-2"

    def test_get_node_boundary_error(self) -> None:
        """越界索引抛出 IndexError。"""
        tree = _make_tree()
        with pytest.raises(IndexError):
            tree.get_node(5, 0, 0)
        with pytest.raises(IndexError):
            tree.get_node(0, 5, 0)
        with pytest.raises(IndexError):
            tree.get_node(0, 0, 5)

    def test_get_node_negative_index_error(self) -> None:
        """负数索引抛出 IndexError。"""
        tree = _make_tree()
        with pytest.raises(IndexError):
            tree.get_node(-1, 0, 0)


# ---------------------------------------------------------------------------
# 测试: 序列化
# ---------------------------------------------------------------------------


class TestSerialization:
    """pickle 序列化测试。"""

    def test_save_load_roundtrip(self) -> None:
        """pickle 序列化后反序列化，数据完整一致。"""
        tree = _make_tree()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            tree.save(path)
            loaded = TreeIndex.load(path)

        # 元数据一致
        assert loaded.metadata.source_path == tree.metadata.source_path
        assert loaded.metadata.embed_dim == tree.metadata.embed_dim

        # 结构一致
        assert len(loaded.roots) == len(tree.roots)
        for orig_l1, load_l1 in zip(tree.roots, loaded.roots):
            assert orig_l1.id == load_l1.id
            assert len(orig_l1.children) == len(load_l1.children)

        # 嵌入一致
        np.testing.assert_array_equal(loaded.l1_embeddings(), tree.l1_embeddings())
        np.testing.assert_array_equal(
            loaded.l3_embeddings_of(0, 1), tree.l3_embeddings_of(0, 1)
        )

    def test_load_nonexistent_file(self) -> None:
        """加载不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            TreeIndex.load("/tmp/nonexistent_tree_index_abc123.pkl")


# ---------------------------------------------------------------------------
# 测试: 空树边界
# ---------------------------------------------------------------------------


class TestEmptyTree:
    """空树边界情况测试。"""

    def test_empty_tree_l1_embeddings(self) -> None:
        """空树的 l1_embeddings() 返回 [0, D]。"""
        tree = TreeIndex(metadata=_make_meta(), roots=[])
        emb = tree.l1_embeddings()
        assert emb.shape == (0, EMBED_DIM)
        assert emb.dtype == np.float32

    def test_empty_tree_get_node_raises(self) -> None:
        """空树访问节点抛出 IndexError。"""
        tree = TreeIndex(metadata=_make_meta(), roots=[])
        with pytest.raises(IndexError):
            tree.get_node(0, 0, 0)

    def test_l2_embeddings_of_boundary(self) -> None:
        """l2_embeddings_of 越界抛出 ValueError。"""
        tree = _make_tree()
        with pytest.raises(ValueError):
            tree.l2_embeddings_of(10)

    def test_l3_embeddings_of_boundary(self) -> None:
        """l3_embeddings_of 越界抛出 ValueError。"""
        tree = _make_tree()
        with pytest.raises(ValueError):
            tree.l3_embeddings_of(0, 10)

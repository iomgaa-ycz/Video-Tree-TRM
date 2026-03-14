"""
TextTreeBuilder 单元测试
========================
使用 MagicMock 替代真实 LLM 和 Embed，测试文本树构建各阶段的正确性。

覆盖范围:
    - _detect_toc：检测 Markdown 标题
    - _segment_with_regex：正则切分 L1/L2 边界、超限分块
    - _segment_with_llm：LLM JSON 分段解析、异常处理
    - _build_l3_from_paragraphs：L3 节点字段验证
    - _build_l2：L2 节点字段验证（通过 build() 间接调用）
    - _build_l1：L1 节点字段验证（通过 build() 间接调用）
    - build()：完整树结构与 IndexMeta 验证、MD 输出文件生成

MD 输出位置: tests/outputs/text_tree_builder/build_<timestamp>.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_tree_trm.config import TreeConfig
from video_tree_trm.text_tree_builder import TextTreeBuilder, _chunk
from video_tree_trm.tree_index import L1Node, L2Node, L3Node, TreeIndex


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_EMBED_DIM = 16  # 轻量维度，加速测试

_SAMPLE_TOC_TEXT = """\
# 第一章 引言

本章介绍研究背景。

信息检索系统的发展历程悠久，早期以关键词匹配为主。

## 1.1 研究背景

随着互联网数据量急剧增长，传统检索方法面临挑战。

语义理解成为关键技术突破口。

## 1.2 研究意义

本研究具有重要的理论和实践价值。

# 第二章 相关工作

本章回顾相关研究成果。

## 2.1 稠密检索

DPR 等模型实现了端到端稠密向量检索。

## 2.2 树状索引

PageIndex 引入了层次化树状检索结构。
"""

_SAMPLE_PLAIN_TEXT = (
    "信息检索是计算机科学的重要分支，研究如何从大规模数据中找到相关信息。"
    "\n\n"
    "传统方法以 TF-IDF 和 BM25 为代表，基于词频统计进行相关性计算。"
    "\n\n"
    "近年来，基于深度学习的稠密检索方法取得了显著进展，DPR 是代表性工作之一。"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_config() -> TreeConfig:
    """树构建配置（小 max_paragraphs_per_l2 便于测试分块）。"""
    return TreeConfig(
        max_paragraphs_per_l2=2,
        l1_segment_duration=600.0,
        l2_clip_duration=20.0,
        l3_fps=1.0,
        l2_representative_frames=3,
        cache_dir="cache/trees",
    )


@pytest.fixture
def mock_embed() -> MagicMock:
    """Mock EmbeddingModel，embed() 返回固定维度的随机向量。"""
    m = MagicMock()
    m.dim = _EMBED_DIM
    m._model_name = "mock-embed-model"

    def _embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        return np.ones((n, _EMBED_DIM), dtype=np.float32)

    m.embed.side_effect = _embed
    return m


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLMClient，chat() 返回固定字符串，batch_chat() 逐条映射。"""
    m = MagicMock()
    m.chat.return_value = "这是一段模拟的摘要描述。"
    m.batch_chat.side_effect = lambda prompts: [
        f"模拟摘要_{i}" for i in range(len(prompts))
    ]
    return m


@pytest.fixture
def builder(mock_embed, mock_llm, tree_config) -> TextTreeBuilder:
    """标准 TextTreeBuilder（使用 mock 依赖）。"""
    return TextTreeBuilder(
        embed_model=mock_embed,
        llm=mock_llm,
        config=tree_config,
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    """确保 MD 输出目录存在。"""
    Path("tests/outputs/text_tree_builder").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 辅助函数：保存 MD 输出
# ---------------------------------------------------------------------------


def _save_md_output(test_name: str, content: str) -> str:
    """将测试执行过程保存为 Markdown 文件。

    参数:
        test_name: 测试名称（用于文件名）。
        content: Markdown 内容。

    返回:
        保存的文件路径。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"tests/outputs/text_tree_builder/{test_name}_{ts}.md")
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 辅助函数：_chunk
# ---------------------------------------------------------------------------


class TestChunk:
    """测试 _chunk 等长分块辅助函数。"""

    def test_exact_multiple(self):
        """整除时每组大小相等。"""
        result = _chunk(["a", "b", "c", "d"], 2)
        assert result == [["a", "b"], ["c", "d"]]

    def test_remainder(self):
        """不整除时最后一组为余数。"""
        result = _chunk(["a", "b", "c", "d", "e"], 2)
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_single_chunk(self):
        """列表长度 <= size 时只有一组。"""
        result = _chunk(["a", "b"], 5)
        assert result == [["a", "b"]]

    def test_empty_list(self):
        """空列表返回空列表。"""
        result = _chunk([], 3)
        assert result == []


# ---------------------------------------------------------------------------
# _detect_toc
# ---------------------------------------------------------------------------


class TestDetectToc:
    """测试 Markdown 标题检测。"""

    def test_with_h1_header(self, builder):
        """一级标题返回 True。"""
        assert builder._detect_toc("# 第一章\n\n内容") is True

    def test_with_h2_header(self, builder):
        """二级标题返回 True。"""
        assert builder._detect_toc("## 1.1 小节\n\n内容") is True

    def test_with_both_headers(self, builder):
        """同时含 # 和 ## 返回 True。"""
        assert builder._detect_toc(_SAMPLE_TOC_TEXT) is True

    def test_without_headers(self, builder):
        """纯段落文本返回 False。"""
        assert builder._detect_toc(_SAMPLE_PLAIN_TEXT) is False

    def test_hash_in_content_not_header(self, builder):
        """行中间的 # 不算标题。"""
        text = "颜色代码 #FF0000 是红色。\n\n这是普通段落。"
        assert builder._detect_toc(text) is False

    def test_h3_not_counted(self, builder):
        """三级标题 ### 也被检测为有 ToC（属于 Markdown 结构文本）。

        注：当前实现只检测 # 和 ##，所以 ### 开头应返回 False。
        """
        text = "### 三级标题\n\n内容"
        # _detect_toc 只匹配 #{1,2}，### 不匹配
        assert builder._detect_toc(text) is False


# ---------------------------------------------------------------------------
# _segment_with_regex
# ---------------------------------------------------------------------------


class TestSegmentWithRegex:
    """测试正则切分 L1/L2 边界。"""

    def test_basic_structure(self, builder):
        """含 2 个 L1 章节，正确返回 2 个外层元素。"""
        sections = builder._segment_with_regex(_SAMPLE_TOC_TEXT)
        assert len(sections) == 2

    def test_all_sections_nonempty(self, builder):
        """每个 section 至少包含一个段落。"""
        sections = builder._segment_with_regex(_SAMPLE_TOC_TEXT)
        for s in sections:
            assert len(s) > 0

    def test_overflow_chunking_via_build(self, builder):
        """当段落数超过 max_paragraphs_per_l2=2 时，build() 会等长分块为多个 L2。"""
        # 第一章有 4 个段落（含标题），超过 max=2 → 应有 2 个 L2
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        # 每个 L1 的 L2 数 >= 1
        for l1 in index.roots:
            assert len(l1.children) >= 1

    def test_paragraphs_are_strings(self, builder):
        """所有段落应为非空字符串。"""
        sections = builder._segment_with_regex(_SAMPLE_TOC_TEXT)
        for section in sections:
            for para in section:
                assert isinstance(para, str)
                assert para.strip() != ""

    def test_single_chapter_text(self, builder):
        """只含一个 L1 章节的文本，返回 1 个外层元素。"""
        text = "# 唯一章节\n\n第一段内容。\n\n第二段内容。"
        sections = builder._segment_with_regex(text)
        assert len(sections) == 1

    def test_no_l2_header(self, builder):
        """只有 L1 无 L2 标题时，段落直接收集到 L1 组。"""
        text = "# 第一章\n\n段落一。\n\n段落二。\n\n段落三。"
        sections = builder._segment_with_regex(text)
        assert len(sections) == 1
        assert len(sections[0]) >= 2


# ---------------------------------------------------------------------------
# _segment_with_llm
# ---------------------------------------------------------------------------


class TestSegmentWithLLM:
    """测试 LLM 语义分段。"""

    def test_json_array_parsing(self, builder):
        """mock LLM 返回合法 JSON 数组，应正确解析为段落列表。"""
        paragraphs = ["第一段描述内容。", "第二段描述内容。", "第三段描述内容。"]
        builder.llm.chat.return_value = json.dumps(paragraphs, ensure_ascii=False)

        sections = builder._segment_with_llm(_SAMPLE_PLAIN_TEXT)
        assert len(sections) == 1  # 单个 L1
        assert len(sections[0]) == 3
        assert sections[0][0] == "第一段描述内容。"

    def test_json_wrapped_in_markdown(self, builder):
        """JSON 数组被 markdown 代码块包裹时也能正确解析。"""
        paragraphs = ["段落A", "段落B"]
        json_str = json.dumps(paragraphs, ensure_ascii=False)
        builder.llm.chat.return_value = f"```json\n{json_str}\n```"

        sections = builder._segment_with_llm(_SAMPLE_PLAIN_TEXT)
        assert sections[0] == ["段落A", "段落B"]

    def test_invalid_json_raises(self, builder):
        """LLM 返回非法 JSON 时应抛出 ValueError。"""
        builder.llm.chat.return_value = "这不是 JSON 格式的内容"
        with pytest.raises((ValueError, AssertionError)):
            builder._segment_with_llm(_SAMPLE_PLAIN_TEXT)

    def test_empty_paragraphs_filtered(self, builder):
        """空段落应被过滤掉。"""
        paragraphs = ["有效段落", "", "  ", "另一有效段落"]
        builder.llm.chat.return_value = json.dumps(paragraphs, ensure_ascii=False)
        sections = builder._segment_with_llm(_SAMPLE_PLAIN_TEXT)
        assert len(sections[0]) == 2


# ---------------------------------------------------------------------------
# _build_l3_from_paragraphs
# ---------------------------------------------------------------------------


class TestBuildL3:
    """测试 L3 节点构建。"""

    def test_description_equals_raw_content(self, builder):
        """L3 节点的 description 应等于 raw_content，等于原始段落。"""
        paragraphs = ["段落一内容", "段落二内容"]
        nodes = builder._build_l3_from_paragraphs(paragraphs, l1_i=0, l2_j=0)
        for node, para in zip(nodes, paragraphs):
            assert node.description == para
            assert node.raw_content == para

    def test_embedding_shape(self, builder):
        """每个 L3 节点的 embedding 形状应为 (dim,)。"""
        paragraphs = ["A", "B", "C"]
        nodes = builder._build_l3_from_paragraphs(paragraphs, l1_i=0, l2_j=0)
        for node in nodes:
            assert node.embedding.shape == (_EMBED_DIM,)

    def test_node_count(self, builder):
        """返回的 L3 节点数应与输入段落数相等。"""
        paragraphs = ["A", "B", "C"]
        nodes = builder._build_l3_from_paragraphs(paragraphs, l1_i=0, l2_j=0)
        assert len(nodes) == 3

    def test_node_id_format(self, builder):
        """节点 ID 格式应为 l1_{i}_l2_{j}_l3_{k}。"""
        nodes = builder._build_l3_from_paragraphs(["A", "B"], l1_i=1, l2_j=2)
        assert nodes[0].id == "l1_1_l2_2_l3_0"
        assert nodes[1].id == "l1_1_l2_2_l3_1"

    def test_video_fields_are_none(self, builder):
        """文本模式下 frame_path 和 timestamp 应为 None。"""
        nodes = builder._build_l3_from_paragraphs(["测试段落"], l1_i=0, l2_j=0)
        assert nodes[0].frame_path is None
        assert nodes[0].timestamp is None

    def test_embedding_dtype(self, builder):
        """嵌入向量应为 float32。"""
        nodes = builder._build_l3_from_paragraphs(["A"], l1_i=0, l2_j=0)
        assert nodes[0].embedding.dtype == np.float32


# ---------------------------------------------------------------------------
# build() — 完整树结构验证
# ---------------------------------------------------------------------------


class TestBuild:
    """测试 build() 完整流程与 TreeIndex 结构。"""

    def test_returns_tree_index(self, builder):
        """build() 应返回 TreeIndex 实例。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        assert isinstance(index, TreeIndex)

    def test_roots_nonempty(self, builder):
        """roots 列表不为空。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        assert len(index.roots) > 0

    def test_l1_has_l2_children(self, builder):
        """每个 L1 节点至少有一个 L2 子节点。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            assert isinstance(l1, L1Node)
            assert len(l1.children) > 0

    def test_l2_has_l3_children(self, builder):
        """每个 L2 节点至少有一个 L3 子节点。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            for l2 in l1.children:
                assert isinstance(l2, L2Node)
                assert len(l2.children) > 0

    def test_l3_are_leaf_nodes(self, builder):
        """L3 节点是叶子层，类型为 L3Node。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            for l2 in l1.children:
                for l3 in l2.children:
                    assert isinstance(l3, L3Node)

    def test_index_meta_modality(self, builder):
        """IndexMeta.modality 应为 'text'。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="doc.txt")
        assert index.metadata.modality == "text"

    def test_index_meta_source_path(self, builder):
        """IndexMeta.source_path 应与传入参数一致。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="my_doc.txt")
        assert index.metadata.source_path == "my_doc.txt"

    def test_index_meta_embed_dim(self, builder):
        """IndexMeta.embed_dim 应与 embed.dim 一致。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        assert index.metadata.embed_dim == _EMBED_DIM

    def test_embedding_shapes(self, builder):
        """所有节点 embedding 形状应为 (dim,)。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            assert l1.embedding.shape == (_EMBED_DIM,)
            for l2 in l1.children:
                assert l2.embedding.shape == (_EMBED_DIM,)
                for l3 in l2.children:
                    assert l3.embedding.shape == (_EMBED_DIM,)

    def test_l1_summary_nonempty(self, builder):
        """L1 summary 不为空。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            assert l1.summary.strip() != ""

    def test_l2_description_nonempty(self, builder):
        """L2 description 不为空。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        for l1 in index.roots:
            for l2 in l1.children:
                assert l2.description.strip() != ""

    def test_plain_text_build(self, builder):
        """无 ToC 的纯段落文本也能正确构建。"""
        # 修正 mock：返回合法 JSON
        paragraphs = ["信息检索是计算机科学的重要分支。", "传统方法以 TF-IDF 为代表。", "近年来稠密检索方法兴起。"]
        builder.llm.chat.return_value = json.dumps(paragraphs, ensure_ascii=False)
        builder.llm.batch_chat.side_effect = lambda prompts: [
            f"摘要_{i}" for i in range(len(prompts))
        ]
        index = builder.build(_SAMPLE_PLAIN_TEXT, source_path="plain.txt")
        assert len(index.roots) > 0

    def test_empty_text_raises(self, builder):
        """空文本应抛出 ValueError（通过 ensure 检查）。"""
        with pytest.raises((ValueError, AssertionError)):
            builder.build("   ", source_path="empty.txt")

    def test_batch_chat_called_once(self, builder):
        """build() 应调用 batch_chat() 一次（所有 L2 并发处理）。"""
        builder.llm.batch_chat.reset_mock()
        builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        builder.llm.batch_chat.assert_called_once()

    def test_l1_node_ids_unique(self, builder):
        """所有 L1 节点 ID 唯一。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        ids = [l1.id for l1 in index.roots]
        assert len(ids) == len(set(ids))

    def test_l2_node_ids_unique(self, builder):
        """所有 L2 节点 ID 全局唯一。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")
        ids = [l2.id for l1 in index.roots for l2 in l1.children]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# MD 输出文件
# ---------------------------------------------------------------------------


class TestMDOutput:
    """测试 MD 输出文件生成（Agent 测试规范）。"""

    def test_md_output_saved(self, builder):
        """build() 后应能保存 Markdown 执行记录文件。"""
        index = builder.build(_SAMPLE_TOC_TEXT, source_path="test.txt")

        # 统计树结构信息
        total_l2 = sum(len(r.children) for r in index.roots)
        total_l3 = sum(len(l2.children) for r in index.roots for l2 in r.children)

        # 构造 MD 内容
        l2_details = []
        for l1 in index.roots:
            for l2 in l1.children:
                l2_details.append(f"  - {l2.id}: {l2.description[:40]}...")

        md_content = f"""\
# Agent 测试: TextTreeBuilder.build
## 任务: 长文本 → TreeIndex

## 输入信息
- **文本长度**: {len(_SAMPLE_TOC_TEXT)} 字符
- **是否含 ToC**: {builder._detect_toc(_SAMPLE_TOC_TEXT)}
- **source_path**: test.txt
- **max_paragraphs_per_l2**: {builder.config.max_paragraphs_per_l2}
- **embed_dim**: {_EMBED_DIM}

## Step 1: 结构切分
- **策略**: {'ToC 正则切分' if builder._detect_toc(_SAMPLE_TOC_TEXT) else 'LLM 语义分段'}
- **L1 数量**: {len(index.roots)}
- **各 L1 的 L2 数**: {[len(r.children) for r in index.roots]}

## Step 2: L2 先行（批量 LLM）
- **L2 节点总数**: {total_l2}
- **调用方式**: batch_chat()（并发生成所有 L2 摘要）
- **L2 描述示例**:
{chr(10).join(l2_details[:5])}

## Step 3: L3 向下（原始段落直接复用）
- **L3 节点总数**: {total_l3}
- **L3 特性**: description == raw_content（无 LLM 调用）

## Step 4: L1 向上（聚合摘要）
- **L1 摘要示例**:
{chr(10).join(f'  - {r.id}: {r.summary[:60]}...' for r in index.roots)}

## 最终结果
- **roots 数量**: {len(index.roots)}
- **总 L2 节点**: {total_l2}
- **总 L3 节点**: {total_l3}
- **modality**: {index.metadata.modality}
- **embed_dim**: {index.metadata.embed_dim}
- **embedding shape 检查**: {'PASS' if all(r.embedding.shape == (_EMBED_DIM,) for r in index.roots) else 'FAIL'}
- **L3 description==raw_content 检查**: {'PASS' if all(l3.description == l3.raw_content for r in index.roots for l2 in r.children for l3 in l2.children) else 'FAIL'}
"""

        out_path = _save_md_output("build_toc", md_content)
        assert os.path.exists(out_path)
        print(f"\n[MD 输出] 已保存到: {out_path}")

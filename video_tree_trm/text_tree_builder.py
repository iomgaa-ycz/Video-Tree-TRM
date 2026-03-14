"""
文本树构建模块
==============
将长文本通过 L2 轴心策略转化为三层 TreeIndex。

构建策略::

    Step 1: _segment_text  — 结构切分，确定 L1/L2 边界
    Step 2: L2 先行        — 从原始内容独立生成 L2 摘要（batch_chat 并发）
    Step 3: L3 向下        — 原始段落文本直接作为 L3，无需二次生成
    Step 4: L1 向上        — 聚合 L2 描述，生成 L1 粗粒度摘要
    Step 5: 组装 TreeIndex

L2 轴心策略解决了循环依赖：
    - L2 描述不依赖 L3，从原始段落直接生成
    - L3 直接使用原始段落文本，不调用 LLM
    - L1 聚合 L2 描述，保证完整覆盖

使用方式::

    builder = TextTreeBuilder(embed_model, llm_client, config.tree)
    index = builder.build(text, source_path="docs/my_doc.txt")
    index.save("cache/my_doc.pkl")
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List, Tuple

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

_L2_PROMPT = (
    "用1-2句话描述以下段落的核心内容，与同级小节形成区分:\n\n{text}"
)

_L1_PROMPT = (
    "用2-3句话总结以下小节的核心内容:\n\n{l2_descriptions}"
)

_SEG_PROMPT = (
    "将以下文本分成若干语义段落，每段为完整语义单元。\n"
    "只返回 JSON 数组，格式: [\"段落1\", \"段落2\", ...]，不要其他内容。\n"
    "文本:\n\n{text}"
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _chunk(lst: List[str], size: int) -> List[List[str]]:
    """将列表等长分块（固定步长，无重叠）。

    参数:
        lst: 待分块的列表。
        size: 每块的最大长度。

    返回:
        分块后的列表，每个元素为一个子列表。
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class TextTreeBuilder:
    """文本模态树构建器。

    将长文本通过 L2 轴心策略（先构建 L2，再向下扩展 L3，向上聚合 L1）
    转化为三层 TreeIndex。节点 embedding 均为 None（由 Pipeline.embed_all 延迟填充）。

    属性:
        llm: LLM 客户端。
        config: 树构建配置。
    """

    def __init__(
        self,
        llm: LLMClient,
        config: TreeConfig,
    ) -> None:
        """初始化文本树构建器。

        参数:
            llm: 已初始化的 LLM 客户端（LLMClient）。
            config: 树构建配置（TreeConfig），关键字段 max_paragraphs_per_l2。

        实现细节:
            构建器不持有 EmbeddingModel，所有 embedding 延迟到检索阶段由 Pipeline 统一计算。
        """
        self.llm = llm
        self.config = config

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build(self, text: str, source_path: str) -> TreeIndex:
        """将长文本构建为三层 TreeIndex。

        参数:
            text: 输入长文本（UTF-8 字符串）。
            source_path: 原始文件路径，写入 IndexMeta。

        返回:
            三层 TreeIndex 对象。

        实现细节:
            1. _segment_text 切分文本 → List[List[str]]（外层=L1，内层=L2段落组）
            2. 将所有 L2 段落组的 prompt 批量送入 llm.batch_chat()，并发获取摘要
            3. 逐层组装 L3→L2→L1 节点
            4. 构建 TreeIndex 并写入日志
        """
        ensure(bool(text.strip()), "输入文本不能为空")
        log_msg("INFO", "开始构建文本树索引", source_path=source_path)

        # Phase 1: 结构切分
        sections = self._segment_text(text)
        ensure(len(sections) > 0, "文本切分结果为空")
        log_msg(
            "INFO",
            "文本切分完成",
            l1_count=len(sections),
            l2_groups=[len(s) for s in sections],
        )

        # Phase 2: 收集所有 L2 段落组，批量生成摘要（L2 先行）
        all_groups: List[Tuple[int, int, List[str]]] = []
        for i, section_paragraphs in enumerate(sections):
            for j, group in enumerate(
                _chunk(section_paragraphs, self.config.max_paragraphs_per_l2)
            ):
                all_groups.append((i, j, group))

        l2_prompts = [
            _L2_PROMPT.format(text="\n\n".join(group))
            for _, _, group in all_groups
        ]
        l2_descs = self.llm.batch_chat(l2_prompts)
        log_msg("INFO", "L2 摘要生成完成", total_l2=len(l2_descs))

        # Phase 3-4: 按 L1 组装三层节点
        # 构建索引映射：(i, j) → 在 all_groups / l2_descs 中的位置
        group_index: dict = {
            (i, j): idx for idx, (i, j, _) in enumerate(all_groups)
        }

        l1_nodes: List[L1Node] = []
        for i, section_paragraphs in enumerate(sections):
            groups = _chunk(section_paragraphs, self.config.max_paragraphs_per_l2)
            l2_nodes: List[L2Node] = []

            for j, group in enumerate(groups):
                idx = group_index[(i, j)]
                desc = l2_descs[idx]
                l3_nodes = self._build_l3_from_paragraphs(group, i, j)
                l2_node = L2Node(
                    id=f"l1_{i}_l2_{j}",
                    description=desc,
                    embedding=None,
                    time_range=None,
                    children=l3_nodes,
                )
                l2_nodes.append(l2_node)

            l1_node = self._build_l1(l2_nodes, f"l1_{i}")
            l1_nodes.append(l1_node)

        # Phase 5: 组装 TreeIndex（embedding 延迟到 Pipeline.embed_all，此处为 None）
        metadata = IndexMeta(
            source_path=source_path,
            modality="text",
            created_at=datetime.now().isoformat(),
        )
        index = TreeIndex(metadata=metadata, roots=l1_nodes)

        total_l2 = sum(len(r.children) for r in l1_nodes)
        total_l3 = sum(
            len(l2.children) for r in l1_nodes for l2 in r.children
        )
        log_json(
            "text_tree_build",
            {
                "source_path": source_path,
                "l1_count": len(l1_nodes),
                "l2_count": total_l2,
                "l3_count": total_l3,
                "embedded": False,
            },
        )
        log_msg(
            "INFO",
            "文本树索引构建完成",
            l1=len(l1_nodes),
            l2=total_l2,
            l3=total_l3,
        )
        return index

    # ------------------------------------------------------------------
    # 内部方法：切分策略
    # ------------------------------------------------------------------

    def _segment_text(self, text: str) -> List[List[str]]:
        """结构切分长文本为 L1/L2 层次。

        参数:
            text: 输入文本。

        返回:
            sections[i] = [paragraph_1, paragraph_2, ...]
            外层列表 = L1 段（章节），内层列表 = L2 单元（段落组内段落）。

        策略:
            有 Markdown 标题 → 正则解析 #/## 边界
            无 Markdown 标题 → LLM 单次调用语义分段
        """
        if self._detect_toc(text):
            log_msg("INFO", "检测到 Markdown 标题，使用正则切分")
            return self._segment_with_regex(text)
        else:
            log_msg("INFO", "未检测到 Markdown 标题，使用 LLM 语义分段")
            return self._segment_with_llm(text)

    def _detect_toc(self, text: str) -> bool:
        """检测文本是否包含 Markdown 标题（有 ToC 结构）。

        参数:
            text: 输入文本。

        返回:
            True 表示有 # 或 ## 开头的标题行，False 表示无。
        """
        return bool(re.search(r"^#{1,2}\s+\S", text, re.MULTILINE))

    def _segment_with_regex(self, text: str) -> List[List[str]]:
        """通过正则解析 Markdown 标题边界进行结构切分。

        参数:
            text: 含 Markdown 标题的文本。

        返回:
            List[List[str]]，外层=L1章节，内层=该章节下的段落列表。
            若二级标题下段落数超过 max_paragraphs_per_l2，则进一步等长分块。

        实现细节:
            - # 标题 → L1 边界
            - ## 标题 → L2 边界
            - ### 及以下标题视为段落内容，收集到最近 L2 段落组
            - 空段落过滤掉
        """
        lines = text.split("\n")

        sections: List[List[str]] = []       # 外层=L1
        current_section: List[str] = []      # 当前 L1 下的段落（扁平）
        current_para_lines: List[str] = []   # 积累段落文本行

        def _flush_para() -> None:
            """将当前积累的行合并为一个段落加入 current_section。"""
            para = "\n".join(current_para_lines).strip()
            if para:
                current_section.append(para)
            current_para_lines.clear()

        def _flush_section() -> None:
            """将当前 section 保存，重置。"""
            _flush_para()
            if current_section:
                sections.append(list(current_section))
            current_section.clear()

        for line in lines:
            h1_match = re.match(r"^#\s+(.+)", line)
            h2_match = re.match(r"^##\s+(.+)", line)

            if h1_match:
                # L1 边界：保存当前 section
                _flush_section()
                # 将 H1 标题本身作为第一段落（可选：也可忽略标题行）
                title = h1_match.group(1).strip()
                if title:
                    current_section.append(title)
            elif h2_match:
                # L2 边界：只冲刷当前段落，不切换 section
                _flush_para()
                title = h2_match.group(1).strip()
                if title:
                    current_section.append(title)
            else:
                # 普通内容行（含 ###、####、正文段落）
                if line.strip() == "":
                    # 空行触发段落分隔
                    _flush_para()
                else:
                    current_para_lines.append(line)

        _flush_section()

        # 若没有产生任何 section（如文本只有一个 L1），保底处理
        if not sections:
            sections = [self._collect_paragraphs(text)]

        # 对超出 max_paragraphs_per_l2 的段落组不做处理（由 build() 负责分块）
        return sections

    def _segment_with_llm(self, text: str) -> List[List[str]]:
        """通过 LLM 单次调用语义分段无结构文本。

        参数:
            text: 无 Markdown 标题的纯文本。

        返回:
            List[List[str]]，只有一个外层元素（整篇视为单个 L1）。
            内层为 LLM 返回的语义段落列表。

        异常:
            ValueError: LLM 返回的内容无法解析为 JSON 数组。
        """
        prompt = _SEG_PROMPT.format(text=text)
        raw = self.llm.chat(prompt)

        # 尝试从返回结果中提取 JSON 数组
        raw = raw.strip()
        # 提取可能被 markdown 代码块包裹的 JSON
        code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if code_match:
            raw = code_match.group(1)

        ensure(
            raw.startswith("["),
            f"LLM 语义分段返回格式错误，期望 JSON 数组，实际: {raw[:100]}",
        )

        try:
            paragraphs: List[str] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 语义分段 JSON 解析失败: {e}\n原始输出: {raw}") from e

        ensure(isinstance(paragraphs, list), "LLM 返回值不是列表")
        ensure(len(paragraphs) > 0, "LLM 语义分段返回空列表")

        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        log_msg("INFO", "LLM 语义分段完成", paragraph_count=len(paragraphs))

        return [paragraphs]  # 整篇视为单个 L1

    def _collect_paragraphs(self, text: str) -> List[str]:
        """按双换行符切分段落（保底策略）。

        参数:
            text: 输入文本。

        返回:
            非空段落列表。
        """
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paras if paras else [text.strip()]

    # ------------------------------------------------------------------
    # 内部方法：节点构建
    # ------------------------------------------------------------------

    def _build_l2(self, paragraphs: List[str], l2_id: str) -> L2Node:
        """将段落组构建为 L2 节点（含 LLM 摘要和嵌入）。

        参数:
            paragraphs: 该 L2 节点下的段落文本列表。
            l2_id: 节点 ID。

        返回:
            L2Node（children 为空，由调用方填充）。
        """
        ensure(len(paragraphs) > 0, f"L2 节点 {l2_id} 的段落列表为空")
        prompt = _L2_PROMPT.format(text="\n\n".join(paragraphs))
        description = self.llm.chat(prompt)
        return L2Node(
            id=l2_id,
            description=description,
            embedding=None,
            time_range=None,
        )

    def _build_l3_from_paragraphs(
        self,
        paragraphs: List[str],
        l1_i: int,
        l2_j: int,
    ) -> List[L3Node]:
        """将段落列表批量构建为 L3 节点（原始文本直接复用，不调用 LLM）。

        参数:
            paragraphs: 段落文本列表。
            l1_i: 父 L1 索引（用于生成 ID）。
            l2_j: 父 L2 索引（用于生成 ID）。

        返回:
            L3Node 列表，description == raw_content == 原始段落文本。

        实现细节:
            使用 embed.embed(paragraphs) 批量嵌入，一次调用获取全部向量。
        """
        ensure(len(paragraphs) > 0, f"L3 段落列表为空 (l1={l1_i}, l2={l2_j})")
        nodes: List[L3Node] = []
        for k, para in enumerate(paragraphs):
            nodes.append(
                L3Node(
                    id=f"l1_{l1_i}_l2_{l2_j}_l3_{k}",
                    description=para,
                    embedding=None,
                    raw_content=para,
                    frame_path=None,
                    timestamp=None,
                )
            )
        return nodes

    def _build_l1(self, l2_children: List[L2Node], l1_id: str) -> L1Node:
        """聚合 L2 描述，构建 L1 节点（含 LLM 摘要和嵌入）。

        参数:
            l2_children: 该 L1 节点下的所有 L2 节点。
            l1_id: 节点 ID。

        返回:
            L1Node（children 已由调用方赋值，或在此赋值）。

        实现细节:
            将所有 L2 描述拼接，用序号标注后送入 LLM 生成 2-3 句摘要。
        """
        ensure(len(l2_children) > 0, f"L1 节点 {l1_id} 没有 L2 子节点")
        l2_descriptions = "\n".join(
            f"{idx + 1}. {node.description}"
            for idx, node in enumerate(l2_children)
        )
        prompt = _L1_PROMPT.format(l2_descriptions=l2_descriptions)
        summary = self.llm.chat(prompt)
        return L1Node(
            id=l1_id,
            summary=summary,
            embedding=None,
            time_range=None,
            children=l2_children,
        )

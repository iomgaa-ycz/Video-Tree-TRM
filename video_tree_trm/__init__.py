"""
Video-Tree-TRM 核心包
=====================
结合 TRM 多层对话探索能力与 PageIndex 树状检索能力的新型 Video RAG。
"""

from video_tree_trm.config import Config
from video_tree_trm.embeddings import EmbeddingModel
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.text_tree_builder import TextTreeBuilder
from video_tree_trm.tree_index import (
    IndexMeta,
    L1Node,
    L2Node,
    L3Node,
    TreeIndex,
)

__all__ = [
    "Config",
    "EmbeddingModel",
    "LLMClient",
    "TextTreeBuilder",
    "IndexMeta",
    "L1Node",
    "L2Node",
    "L3Node",
    "TreeIndex",
]

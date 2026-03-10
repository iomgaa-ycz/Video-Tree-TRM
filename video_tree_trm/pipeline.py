"""
端到端推理管线
==============
串联 预处理 → 检索 → 生成 的完整推理流程。
提供 ``build_index()`` 和 ``query()`` 两个高层接口。

使用方式::

    from video_tree_trm.config import Config
    from video_tree_trm.pipeline import Pipeline

    cfg = Config.load("config/default.yaml")
    pipeline = Pipeline(cfg)

    # 构建（或从缓存加载）树索引
    tree = pipeline.build_index("data/my_doc.txt", modality="text")

    # 问答
    answer = pipeline.query("文档的主要结论是什么？", tree)
    print(answer)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from utils.logger_system import ensure, log_msg
from video_tree_trm.answer_generator import AnswerGenerator
from video_tree_trm.config import Config
from video_tree_trm.embeddings import EmbeddingModel
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.recursive_retriever import RecursiveRetriever
from video_tree_trm.text_tree_builder import TextTreeBuilder
from video_tree_trm.tree_index import TreeIndex
from video_tree_trm.video_tree_builder import VideoTreeBuilder


class Pipeline:
    """端到端推理管线（预处理 → 检索 → 生成）。

    将所有子模块按配置串联，对外暴露两个接口：
    - ``build_index()``: 从原始文件构建 TreeIndex，支持磁盘缓存。
    - ``query()``: 对已有 TreeIndex 执行问答，返回生成答案字符串。

    属性:
        config: 全局配置对象。
        embed_model: 文本嵌入模型（冻结）。
        llm: 文本大语言模型客户端。
        vlm: 视觉语言模型客户端。
        retriever: TRM 递归检索器（eval 模式）。
        generator: 答案生成器。
    """

    def __init__(self, config: Config) -> None:
        """初始化端到端推理管线。

        参数:
            config: 通过 ``Config.load()`` 加载的全局配置对象。

        实现细节:
            - 若 ``config.retriever.checkpoint`` 非 None，加载预训练权重。
            - 检索器始终切换到 eval 模式（关闭 Dropout 等训练行为）。
        """
        self.config = config

        # Phase 1: 初始化各子模块
        self.embed_model = EmbeddingModel(config.embed)
        self.llm = LLMClient(config.llm)
        self.vlm = LLMClient(config.vlm)
        self.retriever = RecursiveRetriever(config.retriever)

        # Phase 2: 可选加载检查点
        if config.retriever.checkpoint:
            ensure(
                os.path.isfile(config.retriever.checkpoint),
                f"检查点文件不存在: {config.retriever.checkpoint}",
            )
            state_dict = torch.load(config.retriever.checkpoint, map_location="cpu")
            self.retriever.load_state_dict(state_dict)
            log_msg(
                "INFO",
                "检索器权重已加载",
                checkpoint=config.retriever.checkpoint,
            )

        self.retriever.eval()
        self.generator = AnswerGenerator(self.llm, self.vlm)

        log_msg(
            "INFO",
            "Pipeline 初始化完成",
            modality_embed=config.embed.model_name,
            has_checkpoint=bool(config.retriever.checkpoint),
        )

    def build_index(self, source_path: str, modality: str) -> TreeIndex:
        """构建并缓存 TreeIndex。

        参数:
            source_path: 原始文件路径（文本文件或视频文件）。
            modality:    模态类型，"text" 或 "video"。

        返回:
            构建完成的 TreeIndex 对象。

        实现细节:
            - 缓存路径: ``{cache_dir}/{stem}_{modality}.pkl``。
            - 缓存命中时直接反序列化返回，不重新构建。
            - 缓存未命中时构建后序列化保存到磁盘。
            - 文本模式读取文件内容后调用 TextTreeBuilder；
              视频模式直接将路径传给 VideoTreeBuilder。
        """
        ensure(
            modality in ("text", "video"),
            f"modality 须为 'text' 或 'video'，实际={modality}",
        )

        # Phase 1: 缓存路径计算
        stem = Path(source_path).stem
        cache_dir = Path(self.config.tree.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"{stem}_{modality}.pkl")

        if os.path.isfile(cache_path):
            log_msg("INFO", "缓存命中，直接加载 TreeIndex", cache_path=cache_path)
            return TreeIndex.load(cache_path)

        # Phase 2: 构建树索引
        log_msg(
            "INFO",
            "缓存未命中，开始构建 TreeIndex",
            source_path=source_path,
            modality=modality,
        )

        if modality == "text":
            with open(source_path, encoding="utf-8") as f:
                text = f.read()
            builder = TextTreeBuilder(self.embed_model, self.llm, self.config.tree)
            tree = builder.build(text, source_path)
        else:
            builder = VideoTreeBuilder(self.embed_model, self.vlm, self.config.tree)
            tree = builder.build(source_path)

        # Phase 3: 保存缓存
        tree.save(cache_path)
        log_msg("INFO", "TreeIndex 已缓存", cache_path=cache_path)

        return tree

    def query(self, question: str, tree: TreeIndex) -> str:
        """对已有 TreeIndex 执行问答，返回生成答案。

        参数:
            question: 用户查询字符串。
            tree:     预构建的 TreeIndex（通过 build_index 或外部加载）。

        返回:
            生成的答案字符串。

        实现细节:
            - embed_tensor(question) 返回 [1, D] Tensor（单文本时 N=1）。
            - 检索器在 torch.no_grad() 上下文中运行，避免梯度计算开销。
            - generator.generate 接收 result["paths"]（List[Tuple[int,int,int]]）。
        """
        # Phase 1: 嵌入查询
        q: torch.Tensor = self.embed_model.embed_tensor(question)  # [1, D]

        # Phase 2: 递归检索
        with torch.no_grad():
            result = self.retriever(q, tree)

        log_msg(
            "INFO",
            "检索完成",
            num_rounds=result["num_rounds"],
            num_paths=len(result["paths"]),
            question=question[:50],
        )

        # Phase 3: 生成答案
        return self.generator.generate(question, result["paths"], tree)

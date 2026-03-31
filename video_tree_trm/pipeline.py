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
from typing import Optional

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

        # Phase 1: 初始化各子模块（embed_model 懒加载，仅 query/embed 时触发）
        self._embed_model: Optional[EmbeddingModel] = None
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

    @property
    def embed_model(self) -> EmbeddingModel:
        """懒加载 EmbeddingModel，仅在首次访问时初始化（index 阶段不触发）。"""
        if self._embed_model is None:
            log_msg("INFO", "懒加载 EmbeddingModel", model=self.config.embed.model_name)
            self._embed_model = EmbeddingModel(self.config.embed)
        return self._embed_model

    def build_index(self, source_path: str, modality: str) -> TreeIndex:
        """构建并缓存 TreeIndex（JSON 格式，含 embedding）。

        参数:
            source_path: 原始文件路径（文本文件或视频文件）。
            modality:    模态类型，"text" 或 "video"。

        返回:
            构建完成的 TreeIndex 对象（已 embed）。

        实现细节:
            - 缓存路径: ``{cache_dir}/{stem}_{modality}.json``。
            - 缓存命中时直接反序列化返回（自动恢复 embedding 若有）。
            - 缓存未命中时调用 VLM 生成描述文本，执行 embedding，保存为 JSON。
        """
        ensure(
            modality in ("text", "video"),
            f"modality 须为 'text' 或 'video'，实际={modality}",
        )

        # Phase 1: 缓存路径计算
        stem = Path(source_path).stem
        cache_dir = Path(self.config.tree.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"{stem}_{modality}.json")

        if os.path.isfile(cache_path):
            log_msg("INFO", "缓存命中，直接加载 TreeIndex", cache_path=cache_path)
            tree = TreeIndex.load_json(cache_path)
            # 若缓存中已有 embedding，直接返回；否则按需 embed
            if tree.is_embedded:
                return tree
            log_msg("INFO", "缓存中无 embedding，开始执行 embed_all")
            self._embed_tree(tree, cache_path=cache_path)
            return tree

        # Phase 2: 构建树索引（纯 VLM 文字描述）
        log_msg(
            "INFO",
            "缓存未命中，开始构建 TreeIndex",
            source_path=source_path,
            modality=modality,
        )

        if modality == "text":
            with open(source_path, encoding="utf-8") as f:
                text = f.read()
            builder = TextTreeBuilder(self.llm, self.config.tree)
            tree = builder.build(text, source_path)
        else:
            builder = VideoTreeBuilder(self.vlm, self.config.tree)
            tree = builder.build(source_path)

        # Phase 3: 执行 embedding 并保存（含 embedding）
        self._embed_tree(tree, cache_path=cache_path)

        return tree

    def _embed_tree(self, tree: TreeIndex, cache_path: Optional[str] = None) -> None:
        """对树的所有节点执行 embedding，可选回写缓存。

        参数:
            tree: 待 embed 的 TreeIndex（embedding=None 的节点）。
            cache_path: 若非 None，embed 完成后回写到此路径（JSON 格式，含 embedding）。

        实现细节:
            调用 TreeIndex.embed_all，传入 EmbeddingModel.embed 作为 embed_fn。
            embed_all 内部按 L2 分组批量处理 L3，减少 API 调用次数。
            若 cache_path 非 None，保存时 include_embedding=True。
        """
        log_msg("INFO", "开始对树执行 embedding")
        tree.embed_all(
            embed_fn=self.embed_model.embed,
            model_name=self.config.embed.model_name,
            embed_dim=self.embed_model.dim,
        )
        if cache_path is not None:
            tree.save_json(cache_path, include_embedding=True)
            log_msg("INFO", "embed_all 完成，缓存已更新（含 embedding）", cache_path=cache_path)
        else:
            log_msg("INFO", "embed_all 完成（仅内存，未写磁盘）")

    def _load_or_build_video_tree(self, video_path: str) -> TreeIndex:
        """根据视频路径优先从缓存加载 TreeIndex，若无缓存则在线构建。

        参数:
            video_path: 视频文件路径或 youtube_id。

        返回:
            加载或构建完成的 TreeIndex 对象。
        """
        # 如果传入的是 youtube_id，尝试拼凑路径
        if not os.path.isfile(video_path):
            video_path_full = os.path.join("data/videomme/videos", f"{video_path}.mp4")
            if os.path.isfile(video_path_full):
                video_path = video_path_full

        return self.build_index(video_path, modality="video")

    def query(
        self,
        question: str,
        tree: TreeIndex | str,
        modality: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> str:
        """执行端到端问答。

        参数:
            question: 用户查询字符串。
            tree:     TreeIndex 对象，或树 JSON 路径，或视频路径。
            modality: 当 tree 为字符串且无法自动推断时，指定模态 ("text" 或 "video")。
            cache_path: 若非 None，embed 完成后回写到此路径。

        返回:
            生成的答案字符串。
        """
        # Phase 0: 处理输入，确保得到 TreeIndex 对象
        if isinstance(tree, str):
            if tree.endswith(".json"):
                log_msg("INFO", "直接从 JSON 路径加载 TreeIndex", path=tree)
                tree_obj = TreeIndex.load_json(tree)
                # 若 cache_path 未指定，使用 tree 的 JSON 路径
                if cache_path is None:
                    cache_path = tree
            elif modality == "video" or tree.endswith(".mp4"):
                log_msg("INFO", "根据视频路径获取 TreeIndex", path=tree)
                tree_obj = self._load_or_build_video_tree(tree)
            else:
                # 默认为文本
                log_msg("INFO", "根据文本路径获取 TreeIndex", path=tree)
                tree_obj = self.build_index(tree, modality="text")
        else:
            tree_obj = tree

        # Phase 1: 确保树已 embed
        if not tree_obj.is_embedded:
            log_msg("INFO", "树尚未 embed，触发 embed_all 并回写缓存", cache_path=cache_path)
            self._embed_tree(tree_obj, cache_path=cache_path)

        # Phase 2: 嵌入查询
        q: torch.Tensor = self.embed_model.embed_tensor(question)  # [1, D]

        # Phase 3: 递归检索
        with torch.no_grad():
            result = self.retriever(q, tree_obj)

        log_msg(
            "INFO",
            "检索完成",
            num_rounds=result["num_rounds"],
            num_paths=len(result["paths"]),
            question=question[:50],
        )

        # Phase 4: 生成答案
        return self.generator.generate(
            question, result["paths"], tree_obj, frame_hits=result.get("frame_hits")
        )

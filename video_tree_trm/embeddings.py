"""
嵌入服务模块
============
封装文本嵌入器，支持本地 sentence-transformers 和远程 OpenAI 兼容 API 两种后端。
提供统一的 ``embed()`` / ``embed_tensor()`` 接口，冻结不训练。

使用方式::

    from video_tree_trm.embeddings import EmbeddingModel
    from video_tree_trm.config import Config

    cfg = Config.load("config/default.yaml")
    model = EmbeddingModel(cfg.embed)
    vecs = model.embed(["你好世界"])      # ndarray [1, D]
    tens = model.embed_tensor(["你好"])   # Tensor [1, D]
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from utils.logger_system import ensure, log_msg
from video_tree_trm.config import EmbedConfig


class EmbeddingModel:
    """文本嵌入器封装（冻结），支持本地和远程双后端。

    本地模式: 使用 sentence-transformers 加载 HuggingFace 模型，本地推理。
    远程模式: 调用 OpenAI 兼容 API（如 GPUStack 上的 qwen3-embedding）。

    属性:
        dim: 嵌入维度 D。
    """

    def __init__(self, config: EmbedConfig) -> None:
        """初始化嵌入模型。

        参数:
            config: 嵌入配置，包含 backend、model_name、embed_dim 等。

        异常:
            ValueError: backend 不是 "local" 或 "remote"。
            ValueError: 远程模式缺少 api_key 或 api_url。
        """
        ensure(
            config.backend in ("local", "remote"),
            f"embed.backend 必须为 'local' 或 'remote'，实际为 '{config.backend}'",
        )
        self._backend = config.backend
        self._dim = config.embed_dim

        if self._backend == "local":
            self._init_local(config)
        else:
            self._init_remote(config)

        log_msg(
            "INFO", "嵌入模型初始化完成", backend=self._backend, model=config.model_name
        )

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_local(self, config: EmbedConfig) -> None:
        """初始化本地 sentence-transformers 模型。

        参数:
            config: 嵌入配置。
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.model_name, device=config.device)
        self._model.eval()
        # 冻结所有参数
        for param in self._model.parameters():
            param.requires_grad = False

        actual_dim = self._model.get_sentence_embedding_dimension()
        ensure(
            actual_dim == self._dim,
            f"模型实际维度 ({actual_dim}) 与配置 embed_dim ({self._dim}) 不一致",
        )

    def _init_remote(self, config: EmbedConfig) -> None:
        """初始化远程 OpenAI 兼容 API 客户端。

        参数:
            config: 嵌入配置。
        """
        ensure(bool(config.api_key), "远程模式必须提供 embed.api_key")
        ensure(bool(config.api_url), "远程模式必须提供 embed.api_url")

        from openai import OpenAI

        self._client = OpenAI(base_url=config.api_url, api_key=config.api_key)
        self._model_name = config.model_name

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """嵌入维度 D。"""
        return self._dim

    def embed(self, texts: Union[str, List[str]]) -> ndarray:
        """文本 → 嵌入向量（L2 归一化）。

        参数:
            texts: 单条文本或文本列表。

        返回:
            [N, D] ndarray，每行 L2 范数为 1.0。单条文本时 N=1。
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._backend == "local":
            return self._embed_local(texts)
        return self._embed_remote(texts)

    def embed_tensor(self, texts: Union[str, List[str]]) -> Tensor:
        """文本 → 嵌入 Tensor（L2 归一化）。

        参数:
            texts: 单条文本或文本列表。

        返回:
            [N, D] torch.Tensor（float32）。
        """
        arr = self.embed(texts)
        return torch.from_numpy(arr).float()

    # ------------------------------------------------------------------
    # 后端实现
    # ------------------------------------------------------------------

    def _embed_local(self, texts: List[str]) -> ndarray:
        """本地 sentence-transformers 推理。

        参数:
            texts: 文本列表。

        返回:
            [N, D] ndarray，L2 归一化。
        """
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        # sentence-transformers encode 返回 ndarray [N, D]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _embed_remote(self, texts: List[str]) -> ndarray:
        """远程 OpenAI 兼容 API 调用。

        参数:
            texts: 文本列表。

        返回:
            [N, D] ndarray，L2 归一化。
        """
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        # 按 index 排序，确保顺序一致
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = np.array(
            [item.embedding for item in sorted_data], dtype=np.float32
        )

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # 避免除零
        embeddings = embeddings / norms

        return embeddings

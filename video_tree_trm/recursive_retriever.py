"""
TRM 递归检索器模块
==================
核心可训练模型。通过 Cross-Attention 节点选择 + MLP 推理 + ACT halt
实现多轮 root-to-leaf 递归检索。

组件层次（由底向上）::

    RMSNorm              — Root Mean Square 归一化
    SwiGLU               — Swish Gate Linear Unit
    ReasoningBlock       — 单层残差 MLP 块 (RMSNorm + SwiGLU)
    ReasoningModule      — L_layers 个 ReasoningBlock 串联
    CrossAttentionSelector — 多头 Cross-Attention 节点选择器（共享跨三层）
    RecursiveRetriever   — 主模型：多轮遍历 + ACT halt

输出数据类::

    RetrievalPath        — 单条 root-to-leaf 路径 + 内容字段
    RetrievalResult      — 完整检索结果

训练 vs 推理行为::

    训练: 固定跑 max_rounds 轮，返回 attn_weights + halt_logits 供 loss 计算
    推理: halt_logit > 0 且至少跑过 1 轮时提前停止

权重共享设计::

    selector / L_level / q_head 跨 L1/L2/L3 三个阶段共享同一套参数。
    z 状态在多轮间保留，累积已检索信息，自动偏离已选区域。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logger_system import ensure, log_msg
from video_tree_trm.config import RetrieverConfig
from video_tree_trm.tree_index import TreeIndex


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    与 LayerNorm 的区别：不减均值，只除以 RMS，计算更轻量。

    属性:
        weight: 可学习的缩放参数，形状 [dim]，初始化为全 1。
        eps: 数值稳定性常量。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """初始化 RMSNorm。

        参数:
            dim: 归一化维度大小。
            eps: 防止除零的小常数。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """对最后一个维度做 RMS 归一化。

        参数:
            x: 输入张量，任意形状，最后维度为 dim。

        返回:
            归一化后的张量，形状与输入相同。
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit。

    实现: W_up(x) → split → SiLU(gate) * val → W_down
    W_up 输出维度为 hidden_dim * 2，split 后各为 hidden_dim。

    属性:
        W_up: 升维投影，[dim → hidden_dim * 2]，无偏置。
        W_down: 降维投影，[hidden_dim → dim]，无偏置。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """初始化 SwiGLU。

        参数:
            dim: 输入/输出维度。
            hidden_dim: 门控隐层维度（实际 W_up 输出为 hidden_dim * 2）。
        """
        super().__init__()
        self.W_up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.W_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状 [..., dim]。

        返回:
            输出张量，形状 [..., dim]。
        """
        gate, val = self.W_up(x).chunk(2, dim=-1)
        return self.W_down(F.silu(gate) * val)


class ReasoningBlock(nn.Module):
    """单层残差推理块。

    结构: output = norm(x + ffn(x))，其中 ffn = SwiGLU。

    属性:
        norm: RMSNorm 归一化层。
        ffn: SwiGLU 前馈网络。
    """

    def __init__(self, dim: int, expansion: float) -> None:
        """初始化 ReasoningBlock。

        参数:
            dim: 输入/输出维度。
            expansion: FFN 隐层扩展比，hidden_dim = int(dim * expansion)。
        """
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播（带残差）。

        参数:
            x: 输入张量，形状 [B, D]。

        返回:
            输出张量，形状 [B, D]。
        """
        return self.norm(x + self.ffn(x))


class ReasoningModule(nn.Module):
    """L-level 多层推理模块（多个 ReasoningBlock 串联）。

    跨 L1/L2/L3 三个层级共享权重，作用于向量（非序列）。

    属性:
        blocks: L_layers 个 ReasoningBlock 的 ModuleList。
    """

    def __init__(self, dim: int, L_layers: int, expansion: float) -> None:
        """初始化 ReasoningModule。

        参数:
            dim: 隐层维度。
            L_layers: ReasoningBlock 堆叠层数。
            expansion: SwiGLU 扩展比。
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [ReasoningBlock(dim, expansion) for _ in range(L_layers)]
        )

    def forward(self, z: Tensor, injection: Tensor) -> Tensor:
        """多层 MLP 推理，注入外部信息。

        参数:
            z: 当前潜在状态，形状 [B, D]。
            injection: 注入信息（selected_info + q），形状 [B, D]。

        返回:
            z_new: 更新后的潜在状态，形状 [B, D]。
        """
        h = z + injection
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# Cross-Attention 节点选择器
# ---------------------------------------------------------------------------


class CrossAttentionSelector(nn.Module):
    """跨层节点选择器（权重共享，用于 L1/L2/L3 三个阶段）。

    使用多头 Cross-Attention 从候选节点中软性选择最相关的节点信息，
    同时记录 argmax 硬索引用于路径追踪。

    属性:
        W_q, W_k, W_v, W_o: 查询/键/值/输出投影矩阵。
        num_heads: 注意力头数。
        head_dim: 每头维度 (embed_dim // num_heads)。
        scale: 缩放因子 (head_dim ** -0.5)。
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """初始化 CrossAttentionSelector。

        参数:
            embed_dim: 嵌入维度，须能被 num_heads 整除。
            num_heads: 注意力头数。

        异常:
            AssertionError: embed_dim 不能被 num_heads 整除时抛出。
        """
        super().__init__()
        ensure(
            embed_dim % num_heads == 0,
            f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除",
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, state: Tensor, candidates: Tensor, k: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """多头 Cross-Attention 节点选择。

        参数:
            state:      当前 q+z 融合状态，形状 [B, D]。
            candidates: 该层候选节点嵌入，形状 [B, N, D]。
            k:          返回 Top-k 个候选。

        返回:
            selected_info: Attention 加权节点信息（软选择，用于状态更新），形状 [B, D]。
            attn_weights:  所有候选节点的归一化权重（用于 loss），形状 [B, N]。
            topk_indices:  得分最高的 k 个节点索引，形状 [B, k]。
            topk_scores:   得分最高的 k 个节点分数，形状 [B, k]。
        """
        B, N, D = candidates.shape

        # Phase 1: 投影 Q/K/V
        Q = self.W_q(state).unsqueeze(1)  # [B, 1, D]
        K = self.W_k(candidates)  # [B, N, D]
        V = self.W_v(candidates)  # [B, N, D]

        # Phase 2: reshape → multi-head
        H, d = self.num_heads, self.head_dim
        Q = Q.view(B, 1, H, d).transpose(1, 2)  # [B, H, 1, d]
        K = K.view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        V = V.view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]

        # Phase 3: Scaled Dot-Product Attention（softmax + weighted sum）
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # [B, H, 1, d]
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, D)
        selected_info = self.W_o(attn_out).squeeze(1)  # [B, D]

        # Phase 4: 注意力权重（各头平均）
        raw_scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = raw_scores.mean(dim=1).squeeze(1).softmax(dim=-1)  # [B, N]

        # Phase 5: Top-k 选取
        k = min(k, N)
        topk_scores, topk_indices = torch.topk(attn_weights, k, dim=-1)

        return selected_info, attn_weights, topk_indices, topk_scores

    def score_frames(self, query_state: Tensor, frame_embs: Tensor) -> Tensor:
        """对叶子层（L3）帧进行打分。

        参数:
            query_state: 当前融合状态 [B, D]。
            frame_embs:  帧嵌入 [B, M, D]。

        返回:
            scores: 每帧的得分 [B, M]。
        """
        _, attn_weights, _, _ = self.forward(query_state, frame_embs, k=1)
        return attn_weights


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------


@dataclass
class FrameHit:
    """关键帧命中信息。

    属性:
        timestamp:  帧时间戳（秒）。
        score:      该帧的检索得分（Cross-Attention 权重）。
        frame_path: 帧图像文件路径。
        l3_id:      所属 L3 节点 ID。
        l2_id:      所属 L2 节点 ID。
        l1_id:      所属 L1 节点 ID。
    """

    timestamp: float
    score: float
    frame_path: str
    l3_id: str
    l2_id: str
    l1_id: str


@dataclass
class RetrievalPath:
    """单条 root-to-leaf 检索路径及其内容。

    属性:
        k1, k2, k3: 层级索引。
        score:      路径综合得分。
        l1_summary, l2_description, l3_description: 文本描述。
        raw_content: 文本内容（文本模态）。
        frame_path:  帧路径（视频模态）。
        timestamp:   帧时间戳（视频模态）。
    """

    k1: int
    k2: int
    k3: int
    score: float
    l1_summary: str
    l2_description: str
    l3_description: str
    raw_content: Optional[str] = None
    frame_path: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class RetrievalResult:
    """检索器完整输出。"""

    query: str
    paths: List[RetrievalPath]
    frame_hits: List[FrameHit]
    num_rounds: int
    z_final: Tensor  # [B, D]


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------


class RecursiveRetriever(nn.Module):
    """TRM 递归检索器主模型。

    通过 Cross-Attention 节点选择 + MLP 推理 + ACT halt 机制实现
    多轮 root-to-leaf 递归检索。三个可训练组件（selector/L_level/q_head）
    跨 L1/L2/L3 三层共享权重。

    属性:
        selector: CrossAttentionSelector，跨层共享。
        L_level: ReasoningModule，跨层共享。
        q_head: Linear(D, 1)，ACT halt 决策头。
        L_cycles: 每层内 L_level 推理迭代次数。
        max_rounds: ACT 最大检索轮次。
    """

    def __init__(self, config: RetrieverConfig) -> None:
        """初始化 RecursiveRetriever。"""
        super().__init__()
        self.config = config
        self.selector = CrossAttentionSelector(config.embed_dim, config.num_heads)
        self.L_level = ReasoningModule(
            config.embed_dim, config.L_layers, config.ffn_expansion
        )
        self.q_head = nn.Linear(config.embed_dim, 1)
        self.L_cycles = config.L_cycles
        self.max_rounds = config.max_rounds

        # 阈值判定：如果最大分数低于此值，视为“未找到”
        self.low_conf_threshold = 0.1

        # 初始化 q_head bias 为 -5（sigmoid(-5) ≈ 0.007，倾向继续）
        with torch.no_grad():
            self.q_head.bias.fill_(-5.0)

        log_msg(
            "INFO",
            "RecursiveRetriever 初始化完成",
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            k_l1=config.k_l1,
            k_l2=config.k_l2,
            k_l3=config.k_l3,
            max_paths=config.max_paths,
        )

    def forward(
        self,
        q: Tensor,
        tree: TreeIndex,
        return_internals: bool = False,
    ) -> Dict[str, Any]:
        """多轮多路径检索。"""
        ensure(q.dim() == 2, f"q 应为 [B, D] 张量，实际 shape={q.shape}")

        z = q.clone()  # [B, D]
        all_paths: List[RetrievalPath] = []
        all_frame_hits: List[FrameHit] = []
        attn_weights_all: List[Tensor] = []
        halt_logits_all: List[Tensor] = []

        for round_idx in range(self.max_rounds):
            # 执行多路径遍历
            round_paths, round_frame_hits, z, step_attns = self._traverse_multi_path(q, z, tree)
            
            all_paths.extend(round_paths)
            all_frame_hits.extend(round_frame_hits)
            attn_weights_all.extend(step_attns)

            # ACT halt 决策
            halt_logit = self.q_head(z)
            halt_logits_all.append(halt_logit)

            # 推理模式下：至少走 1 轮，且没有正在回退/探索的需求时停止
            if not self.training and halt_logit.item() > 0 and round_idx > 0:
                log_msg("INFO", "ACT halt 触发", round_idx=round_idx)
                break

        # 去重并限额（按 score 排序）
        all_paths = sorted(all_paths, key=lambda x: x.score, reverse=True)[:self.config.max_paths]
        # Frame hits 按时间戳去重，保留最高分
        unique_hits: Dict[float, FrameHit] = {}
        for hit in all_frame_hits:
            if hit.timestamp not in unique_hits or hit.score > unique_hits[hit.timestamp].score:
                unique_hits[hit.timestamp] = hit
        all_frame_hits = sorted(unique_hits.values(), key=lambda x: x.score, reverse=True)[:20]

        result: Dict[str, Any] = {
            "paths": all_paths,
            "frame_hits": all_frame_hits,
            "num_rounds": round_idx + 1,
            "z_final": z,
        }
        if return_internals:
            result["attn_weights_per_step"] = attn_weights_all
            result["halt_logits"] = halt_logits_all

        return result

    def _traverse_multi_path(
        self, q: Tensor, z: Tensor, tree: TreeIndex
    ) -> Tuple[List[RetrievalPath], List[FrameHit], Tensor, List[Tensor]]:
        """执行带 Top-k 和回退机制的单轮多路径遍历。"""
        step_attns: List[Tensor] = []
        paths: List[RetrievalPath] = []
        frame_hits: List[FrameHit] = []

        # 1. L1 层级
        M_L1 = torch.tensor(tree.l1_embeddings(), dtype=torch.float32, device=q.device).unsqueeze(0)
        selected_info_l1, attn_w1, topk_idx_l1, topk_scores_l1 = self.selector(q + z, M_L1, k=self.config.k_l1)
        step_attns.append(attn_w1)

        # 2. 遍历 L1 候选
        z_updated_list = []
        for i in range(topk_idx_l1.shape[1]):
            k1 = topk_idx_l1[0, i].item()
            s1 = topk_scores_l1[0, i].item()
            
            z_path_l1 = z + selected_info_l1
            for _ in range(self.L_cycles):
                z_path_l1 = self.L_level(z_path_l1, selected_info_l1 + q)

            # L2 层级
            M_L2 = torch.tensor(tree.l2_embeddings_of(k1), dtype=torch.float32, device=q.device).unsqueeze(0)
            selected_info_l2, attn_w2, topk_idx_l2, topk_scores_l2 = self.selector(q + z_path_l1, M_L2, k=self.config.k_l2)
            step_attns.append(attn_w2)

            # 3. 遍历 L2 候选
            for j in range(topk_idx_l2.shape[1]):
                k2 = topk_idx_l2[0, j].item()
                s2 = topk_scores_l2[0, j].item()
                
                z_path_l2 = z_path_l1 + selected_info_l2
                for _ in range(self.L_cycles):
                    z_path_l2 = self.L_level(z_path_l2, selected_info_l2 + q)

                # L3 层级
                M_L3 = torch.tensor(tree.l3_embeddings_of(k1, k2), dtype=torch.float32, device=q.device).unsqueeze(0)
                selected_info_l3, attn_w3, topk_idx_l3, topk_scores_l3 = self.selector(q + z_path_l2, M_L3, k=self.config.k_l3)
                step_attns.append(attn_w3)

                # 4. 回退/探索逻辑：如果 L3 最高分太低，尝试在当前 L2 下扩大搜索范围 (k_l3 * 2)
                if topk_scores_l3[0, 0] < self.low_conf_threshold:
                    log_msg("INFO", "L3 置信度低，尝试扩大该节点下的搜索范围", k2=k2, score=topk_scores_l3[0, 0].item())
                    _, _, topk_idx_l3, topk_scores_l3 = self.selector(q + z_path_l2, M_L3, k=self.config.k_l3 * 2)

                for l in range(topk_idx_l3.shape[1]):
                    k3 = topk_idx_l3[0, l].item()
                    s3 = topk_scores_l3[0, l].item()
                    
                    total_score = s1 * s2 * s3
                    node = tree.get_node(k1, k2, k3)
                    
                    path = RetrievalPath(
                        k1=k1, k2=k2, k3=k3, score=total_score,
                        l1_summary=tree.roots[k1].summary,
                        l2_description=tree.roots[k1].children[k2].description,
                        l3_description=node.description,
                        raw_content=getattr(node, "raw_content", None),
                        frame_path=getattr(node, "frame_path", None),
                        timestamp=getattr(node, "timestamp", None)
                    )
                    paths.append(path)
                    
                    if path.frame_path:
                        frame_hits.append(FrameHit(
                            timestamp=path.timestamp, score=total_score,
                            frame_path=path.frame_path,
                            l3_id=node.id, l2_id=tree.roots[k1].children[k2].id,
                            l1_id=tree.roots[k1].id
                        ))
                
                z_updated_list.append(z_path_l2 + selected_info_l3)

        # 更新全局 z
        if z_updated_list:
            z = torch.stack(z_updated_list).mean(dim=0)
            for _ in range(self.L_cycles):
                z = self.L_level(z, q)

        return paths, frame_hits, z, step_attns

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
        self, state: Tensor, candidates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """多头 Cross-Attention 节点选择。

        参数:
            state:      当前 q+z 融合状态，形状 [B, D]。
            candidates: 该层候选节点嵌入，形状 [B, N, D]。

        返回:
            selected_info: Attention 加权节点信息（可微），形状 [B, D]。
            attn_weights:  归一化注意力权重（用于 nav loss），形状 [B, N]。
            selected_idx:  argmax 节点索引（用于路径记录），形状 [B]。

        实现细节:
            - Q 来自 state（查询），K/V 来自 candidates（候选节点）
            - 注意力权重取各头平均后 softmax，用于 NavigationLoss
            - selected_idx 为 argmax，不参与梯度传播
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

        # Phase 4: 注意力权重（各头平均，用于 loss 和可解释性）
        raw_scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = raw_scores.mean(dim=1).squeeze(1).softmax(dim=-1)  # [B, N]
        selected_idx = attn_weights.argmax(dim=-1)  # [B]

        return selected_info, attn_weights, selected_idx


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------


@dataclass
class RetrievalPath:
    """单条 root-to-leaf 检索路径及其内容。

    属性:
        k1: L1 节点索引。
        k2: L2 节点索引（相对于 k1 下的子节点）。
        k3: L3 节点索引（相对于 k2 下的子节点）。
        l1_summary: L1 节点摘要文本。
        l2_description: L2 节点描述文本。
        l3_description: L3 节点描述文本。
        raw_content: 原始文本内容（文本模式），视频模式为 None。
        frame_path: 帧图像路径（视频模式），文本模式为 None。
        timestamp: 帧时间戳秒数（视频模式），文本模式为 None。
    """

    k1: int
    k2: int
    k3: int
    l1_summary: str
    l2_description: str
    l3_description: str
    raw_content: Optional[str]
    frame_path: Optional[str]
    timestamp: Optional[float]


@dataclass
class RetrievalResult:
    """检索器完整输出。

    属性:
        query: 原始查询字符串。
        paths: 多轮检索收集的 RetrievalPath 列表。
        num_rounds: 实际检索轮次（≤ max_rounds）。
        z_final: 最终潜在状态，形状 [D]，float32 numpy 数组。
    """

    query: str
    paths: List[RetrievalPath]
    num_rounds: int
    z_final: np.ndarray  # [D] float32


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
        """初始化 RecursiveRetriever。

        参数:
            config: RetrieverConfig，包含 embed_dim/num_heads/L_layers/
                    L_cycles/max_rounds/ffn_expansion 等参数。

        实现细节:
            q_head.bias 初始化为 -5.0，使 sigmoid ≈ 0（倾向"继续"），
            避免模型在训练初期过早停止。
        """
        super().__init__()
        self.selector = CrossAttentionSelector(config.embed_dim, config.num_heads)
        self.L_level = ReasoningModule(
            config.embed_dim, config.L_layers, config.ffn_expansion
        )
        self.q_head = nn.Linear(config.embed_dim, 1)
        self.L_cycles = config.L_cycles
        self.max_rounds = config.max_rounds

        # 初始化 q_head bias 为 -5（sigmoid(-5) ≈ 0.007，倾向继续）
        with torch.no_grad():
            self.q_head.bias.fill_(-5.0)

        log_msg(
            "INFO",
            "RecursiveRetriever 初始化完成",
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            L_layers=config.L_layers,
            L_cycles=config.L_cycles,
            max_rounds=config.max_rounds,
        )

    def forward(
        self,
        q: Tensor,
        tree: TreeIndex,
        return_internals: bool = False,
    ) -> Dict[str, Any]:
        """训练/推理统一入口。

        参数:
            q: 查询嵌入，形状 [B, D]，来自冻结的 text_embed。
            tree: 预构建的 TreeIndex。
            return_internals: 为 True 时返回中间状态（供 loss 计算）。

        返回:
            字典，始终包含:
                "paths":      List[Tuple[int, int, int]]，每轮的 (k1, k2, k3)。
                "num_rounds": int，实际检索轮次。
                "z_final":    Tensor [B, D]，最终潜在状态。
            return_internals=True 时额外包含:
                "attn_weights_per_step": List[Tensor]，每步 [B, N]（共 num_rounds×3 项）。
                "halt_logits":           List[Tensor]，每轮 [B, 1]。

        实现细节:
            - 训练模式: 固定跑 max_rounds 轮，不提前停止。
            - 推理模式: halt_logit > 0 且 round_idx > 0 时提前停止。
            - z 状态在轮间保留，累积已检索信息。
        """
        ensure(q.dim() == 2, f"q 应为 [B, D] 张量，实际 shape={q.shape}")

        z = q.clone()  # [B, D]，初始潜在状态 = 查询嵌入
        paths: List[Tuple[int, int, int]] = []
        attn_weights_all: List[Tensor] = []
        halt_logits_all: List[Tensor] = []

        for round_idx in range(self.max_rounds):
            # 单次完整 L1 → L2 → L3 遍历
            path, z, step_attns = self._traverse_one_path(q, z, tree)
            paths.append(path)
            attn_weights_all.extend(step_attns)  # 每轮追加 3 步的 attn_weights

            # ACT halt 决策
            halt_logit = self.q_head(z)  # [B, 1]
            halt_logits_all.append(halt_logit)

            # 推理模式下：至少走 1 轮，halt_logit > 0 时停止
            if not self.training and halt_logit.item() > 0 and round_idx > 0:
                log_msg(
                    "INFO",
                    "ACT halt 触发，提前停止检索",
                    round_idx=round_idx,
                    halt_logit=round(halt_logit.item(), 4),
                )
                break

        result: Dict[str, Any] = {
            "paths": paths,
            "num_rounds": len(paths),
            "z_final": z,
        }
        if return_internals:
            result["attn_weights_per_step"] = attn_weights_all
            result["halt_logits"] = halt_logits_all

        return result

    def _traverse_one_path(
        self,
        q: Tensor,
        z: Tensor,
        tree: TreeIndex,
    ) -> Tuple[Tuple[int, int, int], Tensor, List[Tensor]]:
        """单次 L1 → L2 → L3 三阶段树遍历。

        参数:
            q: 查询嵌入，形状 [B, D]（冻结，轮间不变）。
            z: 当前潜在状态，形状 [B, D]（轮间累积）。
            tree: TreeIndex。

        返回:
            path:       (k1, k2, k3) 整数三元组，路径索引。
            z_new:      更新后的潜在状态，形状 [B, D]。
            step_attns: 三步的 attn_weights 列表，各 [B, N]。

        实现细节:
            - M_L1/M_L2/M_L3 从 TreeIndex 提取后转为与 q 同设备的 Tensor
            - 每层调用 _select_and_reason，更新 z 并记录 attn_weights
        """
        step_attns: List[Tensor] = []

        # Phase 1: L1 粗粒度路由
        M_L1 = torch.tensor(
            tree.l1_embeddings(), dtype=torch.float32, device=q.device
        )  # [N1, D]
        k1, z, attn_w1 = self._select_and_reason(q, z, M_L1.unsqueeze(0))
        step_attns.append(attn_w1)

        # Phase 2: L2 细粒度聚焦（k1 的子节点）
        M_L2 = torch.tensor(
            tree.l2_embeddings_of(k1), dtype=torch.float32, device=q.device
        )  # [N2, D]
        k2, z, attn_w2 = self._select_and_reason(q, z, M_L2.unsqueeze(0))
        step_attns.append(attn_w2)

        # Phase 3: L3 精确定位（k2 的子节点）
        M_L3 = torch.tensor(
            tree.l3_embeddings_of(k1, k2), dtype=torch.float32, device=q.device
        )  # [N3, D]
        k3, z, attn_w3 = self._select_and_reason(q, z, M_L3.unsqueeze(0))
        step_attns.append(attn_w3)

        return (k1, k2, k3), z, step_attns

    def _select_and_reason(
        self,
        q: Tensor,
        z: Tensor,
        M: Tensor,
    ) -> Tuple[int, Tensor, Tensor]:
        """单层 Cross-Attention 选择 + L_cycles 内循环推理。

        参数:
            q: 查询嵌入，形状 [B, D]（不变）。
            z: 当前潜在状态，形状 [B, D]。
            M: 该层候选节点嵌入，形状 [B, N, D]。

        返回:
            k_star:      int，选中节点索引（argmax，非可微）。
            z_new:       更新后的潜在状态，形状 [B, D]。
            attn_weights: 归一化注意力权重，形状 [B, N]（用于 NavigationLoss）。

        实现细节:
            state = q + z（融合查询与当前状态作为 Q 输入）
            z 先通过 CA 软更新，再经 L_cycles 次 MLP 推理
        """
        state = q + z  # [B, D]
        selected_info, attn_weights, selected_idx = self.selector(state, M)

        # 软更新 z（可微）
        z = z + selected_info

        # L_cycles 次 MLP 推理（注入 selected_info + q）
        for _ in range(self.L_cycles):
            z = self.L_level(z, selected_info + q)

        k_star: int = selected_idx.item()
        return k_star, z, attn_weights

"""
损失函数模块
============
实现两阶段训练所需的两个核心损失函数：

- NavigationLoss: 三层 cross-entropy，推动 attn_weights 指向正确节点
- ACTLoss:        Q-learning 二分类，学习最优 halt 决策

使用方式::

    from video_tree_trm.losses import NavigationLoss, ACTLoss, compute_nav_act_loss

    nav_loss_fn = NavigationLoss()
    act_loss_fn = ACTLoss(lambda_step=0.1, gamma=0.9)

    # Phase 1（单轮导航训练）
    result = retriever(q, tree, return_internals=True)
    loss = nav_loss_fn(result["attn_weights_per_step"][:3], gt_path=(0, 1, 2))

    # Phase 2（多轮 ACT 训练）
    out = compute_nav_act_loss(
        result, gt_path, answer_qualities,
        nav_loss_fn, act_loss_fn,
        nav_weight=1.0, act_weight=0.1,
    )
    out["total"].backward()
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logger_system import ensure, log_msg


# ---------------------------------------------------------------------------
# NavigationLoss
# ---------------------------------------------------------------------------


class NavigationLoss(nn.Module):
    """三层 cross-entropy 导航损失。

    推动 CrossAttentionSelector 在 L1/L2/L3 三层分别将注意力权重集中
    到 ground-truth 路径上对应的正确节点。

    注意: 该损失无可学习参数，仅封装计算逻辑以便与 nn.Module 生态一致。
    """

    def forward(
        self,
        attn_weights_list: List[Tensor],
        gt_path: Tuple[int, int, int],
    ) -> Tensor:
        """计算三层导航损失（NLL loss 均值）。

        参数:
            attn_weights_list: 三个 [B, N] Tensor，分别对应 L1/L2/L3 层
                               的 softmax 注意力权重，来自
                               RecursiveRetriever 的 attn_weights_per_step[:3]。
            gt_path: (gt_l1_idx, gt_l2_idx, gt_l3_idx)，各层正确节点的整数索引。

        返回:
            标量 loss，三层 NLL loss 均值。

        实现细节:
            - attn_w.clamp(min=1e-8) 防止 log(0) = -inf 的数值异常。
            - NLL loss = -log(attn_w[gt_idx])，与 cross-entropy 等价
              （因为 attn_w 已经过 softmax 归一化）。
            - target 需 expand 到 batch size B 以兼容任意批大小。
        """
        ensure(
            len(attn_weights_list) == 3,
            f"attn_weights_list 应包含 3 项（L1/L2/L3），实际={len(attn_weights_list)}",
        )

        total_loss = torch.tensor(0.0, device=attn_weights_list[0].device)

        for attn_w, gt_idx in zip(attn_weights_list, gt_path):
            # attn_w: [B, N]
            B = attn_w.shape[0]
            log_probs = attn_w.clamp(min=1e-8).log()  # [B, N]，防止数值下溢
            target = torch.tensor([gt_idx], device=attn_w.device).expand(B)  # [B]
            total_loss = total_loss + F.nll_loss(log_probs, target)

        return total_loss / 3  # 三层平均


# ---------------------------------------------------------------------------
# ACTLoss
# ---------------------------------------------------------------------------


class ACTLoss(nn.Module):
    """ACT halt Q-learning 损失（BCE 版本）。

    将 halt 决策建模为二分类问题：
    若"现在停止"的预期质量 ≥ "继续检索"的预期质量，则目标为 1（应停止）。

    Q-learning 目标设计::

        Q_halt(t)    = answer_quality[t]
        Q_continue(t) = γ * answer_quality[t+1] - λ   (t < n-1)
        Q_continue(t) = answer_quality[t] - λ           (t == n-1，无法继续)

        target = 1.0 if Q_halt ≥ Q_continue else 0.0
        loss_t = BCE(sigmoid(halt_logit[t]), target)

    属性:
        lambda_step: 步数惩罚系数，每多检索一轮的额外代价。
        gamma:       未来质量的折扣因子。
    """

    def __init__(self, lambda_step: float, gamma: float) -> None:
        """初始化 ACTLoss。

        参数:
            lambda_step: 步数惩罚系数（来自 TrainConfig.act_lambda_step）。
            gamma:       折扣因子（来自 TrainConfig.act_gamma）。

        异常:
            ValueError: lambda_step 或 gamma 不合法时抛出。
        """
        super().__init__()
        ensure(lambda_step >= 0, f"lambda_step 须 ≥ 0，实际={lambda_step}")
        ensure(0 < gamma <= 1, f"gamma 须在 (0, 1]，实际={gamma}")
        self.lambda_step = lambda_step
        self.gamma = gamma
        log_msg("INFO", "ACTLoss 初始化", lambda_step=lambda_step, gamma=gamma)

    def forward(
        self,
        halt_logits: List[Tensor],
        answer_qualities: List[float],
    ) -> Tensor:
        """计算多轮 ACT halt 损失（BCE 均值）。

        参数:
            halt_logits:      每轮 [B, 1] Tensor，来自 q_head(z)，
                              对应 RecursiveRetriever 的 halt_logits 列表。
            answer_qualities: 每轮的答案质量分数（0~1），由外部计算后传入
                              （Phase 2 训练中通过 EM/F1 动态估计）。

        返回:
            标量 loss，所有轮次 BCE 均值。

        实现细节:
            - torch.full_like(pred, target) 兼容任意 batch size B。
            - sigmoid 将 halt_logit 映射到 (0,1) 作为 halt 概率。
        """
        ensure(
            len(halt_logits) == len(answer_qualities),
            f"halt_logits 长度 ({len(halt_logits)}) 与 answer_qualities "
            f"长度 ({len(answer_qualities)}) 不一致",
        )
        n = len(halt_logits)
        ensure(n > 0, "halt_logits 不能为空")

        total_loss = torch.tensor(0.0, device=halt_logits[0].device)

        for t in range(n):
            q_halt = answer_qualities[t]

            # Q_continue：有下一轮则折扣未来质量，否则减步数惩罚
            if t < n - 1:
                q_continue = self.gamma * answer_qualities[t + 1] - self.lambda_step
            else:
                q_continue = q_halt - self.lambda_step

            # 二分类目标：halt_q ≥ continue_q 时应停止
            target = 1.0 if q_halt >= q_continue else 0.0

            pred = torch.sigmoid(halt_logits[t])  # [B, 1]
            target_tensor = torch.full_like(pred, target)  # [B, 1]
            total_loss = total_loss + F.binary_cross_entropy(pred, target_tensor)

        return total_loss / n  # 轮次平均


# ---------------------------------------------------------------------------
# 组合加权损失辅助函数
# ---------------------------------------------------------------------------


def compute_nav_act_loss(
    result: Dict[str, Any],
    gt_path: Tuple[int, int, int],
    answer_qualities: List[float],
    nav_loss_fn: NavigationLoss,
    act_loss_fn: ACTLoss,
    nav_weight: float,
    act_weight: float,
) -> Dict[str, Tensor]:
    """组合加权损失入口，供 train.py 调用。

    将 NavigationLoss 和 ACTLoss 按权重加权求和，封装为一个便捷函数。

    参数:
        result:           RecursiveRetriever.forward(return_internals=True) 的返回字典，
                          需包含 "attn_weights_per_step" 和 "halt_logits" 键。
        gt_path:          标注的 ground-truth 路径 (gt_l1, gt_l2, gt_l3)。
        answer_qualities: 每轮答案质量（Phase 1 可传 [0.0] 占位，Phase 2 动态计算）。
        nav_loss_fn:      NavigationLoss 实例。
        act_loss_fn:      ACTLoss 实例。
        nav_weight:       导航损失权重（TrainConfig.nav_loss_weight，默认 1.0）。
        act_weight:       ACT 损失权重（TrainConfig.act_loss_weight，默认 0.1）。

    返回:
        字典 {"total": Tensor, "nav": Tensor, "act": Tensor}，
        total = nav_weight * nav + act_weight * act。

    实现细节:
        - attn_weights_per_step[:3] 仅取第一轮的三步注意力权重用于导航损失。
        - halt_logits 覆盖所有轮次，用于 ACT 损失。
    """
    ensure(
        "attn_weights_per_step" in result,
        "result 须包含 'attn_weights_per_step' 键（需 return_internals=True）",
    )
    ensure(
        "halt_logits" in result,
        "result 须包含 'halt_logits' 键（需 return_internals=True）",
    )

    # Phase 1: 仅取第一轮的 L1/L2/L3 三步注意力权重
    attn_w_first_round: List[Tensor] = result["attn_weights_per_step"][:3]
    halt_logits: List[Tensor] = result["halt_logits"]

    l_nav = nav_loss_fn(attn_w_first_round, gt_path)
    l_act = act_loss_fn(halt_logits, answer_qualities)
    l_total = nav_weight * l_nav + act_weight * l_act

    log_msg(
        "INFO",
        "损失计算完成",
        nav=round(l_nav.item(), 4),
        act=round(l_act.item(), 4),
        total=round(l_total.item(), 4),
    )

    return {"total": l_total, "nav": l_nav, "act": l_act}

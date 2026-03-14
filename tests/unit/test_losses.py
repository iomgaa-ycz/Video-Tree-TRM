"""
test_losses.py — NavigationLoss / ACTLoss / compute_nav_act_loss 单元测试
=========================================================================
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from video_tree_trm.losses import ACTLoss, NavigationLoss, compute_nav_act_loss


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_attn(weights: List[float], B: int = 1) -> Tensor:
    """构造已归一化的注意力权重 Tensor [B, N]。"""
    t = torch.tensor(weights, dtype=torch.float32)
    t = t / t.sum()  # 确保 sum=1（模拟 softmax 输出）
    return t.unsqueeze(0).expand(B, -1)


def _make_halt_logit(value: float) -> Tensor:
    """构造单个 halt logit [1, 1]。"""
    return torch.tensor([[value]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# NavigationLoss 测试
# ---------------------------------------------------------------------------


def test_nav_loss_low_when_correct() -> None:
    """正确节点权重最大时，loss 应较低（-log(max_prob) 最小）。"""
    fn = NavigationLoss()
    # gt_idx=0 对应最高权重 0.9
    attn = _make_attn([0.9, 0.05, 0.05])
    loss = fn([attn, attn, attn], gt_path=(0, 0, 0))
    expected = -math.log(0.9)
    assert abs(loss.item() - expected) < 1e-4, (
        f"loss={loss.item():.4f}, 预期≈{expected:.4f}"
    )


def test_nav_loss_high_when_wrong() -> None:
    """正确节点权重最小时，loss 应高于正确时的 loss。"""
    fn = NavigationLoss()
    correct_attn = _make_attn([0.9, 0.05, 0.05])
    wrong_attn = _make_attn([0.05, 0.05, 0.9])
    # gt_idx=0，但权重仅 0.05
    loss_wrong = fn([wrong_attn, wrong_attn, wrong_attn], gt_path=(0, 0, 0))
    loss_correct = fn([correct_attn, correct_attn, correct_attn], gt_path=(0, 0, 0))
    assert loss_wrong.item() > loss_correct.item(), "错误节点 loss 应高于正确节点"


def test_nav_loss_three_layers_averaged() -> None:
    """三层 loss 应等于各层单独计算的均值（数学等价验证）。"""
    fn = NavigationLoss()
    attn_l1 = _make_attn([0.7, 0.2, 0.1])
    attn_l2 = _make_attn([0.4, 0.6])
    attn_l3 = _make_attn([0.5, 0.3, 0.2])

    combined = fn([attn_l1, attn_l2, attn_l3], gt_path=(0, 1, 2))

    # 手动计算各层
    l1 = F.nll_loss(attn_l1.clamp(min=1e-8).log(), torch.tensor([0]))
    l2 = F.nll_loss(attn_l2.clamp(min=1e-8).log(), torch.tensor([1]))
    l3 = F.nll_loss(attn_l3.clamp(min=1e-8).log(), torch.tensor([2]))
    expected = (l1 + l2 + l3) / 3

    assert torch.allclose(combined, expected, atol=1e-5), (
        f"三层均值不符: combined={combined.item():.5f}, expected={expected.item():.5f}"
    )


def test_nav_loss_clamp_avoids_inf() -> None:
    """某节点权重为 0 时，loss 应为有限值（不产生 inf）。"""
    fn = NavigationLoss()
    # 手动构造含 0 的权重（未经 softmax，但模拟极端情况）
    attn = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    loss = fn([attn, attn, attn], gt_path=(0, 0, 0))
    assert math.isfinite(loss.item()), f"loss 不应为 inf，实际={loss.item()}"


def test_nav_loss_wrong_input_length() -> None:
    """输入不足 3 层时应抛出 ValueError。"""
    fn = NavigationLoss()
    attn = _make_attn([0.5, 0.5])
    with pytest.raises(ValueError):
        fn([attn, attn], gt_path=(0, 0, 0))


# ---------------------------------------------------------------------------
# ACTLoss 测试
# ---------------------------------------------------------------------------


def test_act_loss_should_halt() -> None:
    """Q_halt > Q_continue 时 target=1，pred=sigmoid(正值) 接近 1 → loss 低。"""
    fn = ACTLoss(lambda_step=0.1, gamma=0.9)
    # 单轮：Q_halt=0.8，Q_continue = 0.8 - 0.1 = 0.7 → Q_halt > Q_continue → target=1
    halt_logit = _make_halt_logit(3.0)  # sigmoid(3)≈0.95，接近 target=1
    loss = fn([halt_logit], answer_qualities=[0.8])
    # BCE(0.95, 1.0) 应很小
    assert loss.item() < 0.2, f"应 halt 时 loss 应低，实际={loss.item():.4f}"


def test_act_loss_should_continue() -> None:
    """Q_halt < Q_continue 时 target=0，pred=sigmoid(负值) 接近 0 → loss 低。"""
    fn = ACTLoss(lambda_step=0.1, gamma=0.9)
    # 两轮：t=0 时 Q_halt=0.3，Q_continue=0.9*0.8-0.1=0.62 → Q_halt < Q_continue → target=0
    halt_logit_t0 = _make_halt_logit(-3.0)  # sigmoid(-3)≈0.05，接近 target=0
    halt_logit_t1 = _make_halt_logit(0.0)
    fn([halt_logit_t0, halt_logit_t1], answer_qualities=[0.3, 0.8])
    # 取第一轮的 loss（target=0, pred≈0.05）
    bce_t0 = F.binary_cross_entropy(
        torch.sigmoid(halt_logit_t0),
        torch.zeros_like(halt_logit_t0),
    )
    # 第一轮 loss 应很小
    assert bce_t0.item() < 0.2, f"应继续时 t=0 的 BCE 应低，实际={bce_t0.item():.4f}"


def test_act_loss_last_round() -> None:
    """最后一轮 Q_continue = quality - lambda_step（无法继续）。"""
    lambda_step = 0.1
    fn = ACTLoss(lambda_step=lambda_step, gamma=0.9)
    quality = 0.5
    # 单轮时：Q_halt=0.5，Q_continue=0.5-0.1=0.4 → target=1
    halt_logit = _make_halt_logit(3.0)  # sigmoid(3)≈0.95 → 接近 target=1
    loss = fn([halt_logit], answer_qualities=[quality])
    # Q_halt(0.5) > Q_continue(0.4)，target=1，pred≈0.95 → loss 低
    assert loss.item() < 0.2, f"最后一轮 target=1 时 loss 应低，实际={loss.item():.4f}"


def test_act_loss_averaged() -> None:
    """两轮 loss 应等于各轮 BCE 均值（数学等价验证）。"""
    fn = ACTLoss(lambda_step=0.1, gamma=0.9)
    halt_logits = [_make_halt_logit(1.0), _make_halt_logit(-1.0)]
    qualities = [0.8, 0.6]

    combined = fn(halt_logits, answer_qualities=qualities)

    # 手动计算：
    # t=0: Q_halt=0.8, Q_continue=0.9*0.6-0.1=0.44 → target=1
    target_0 = 1.0
    # t=1: Q_halt=0.6, Q_continue=0.6-0.1=0.5 → target=1
    target_1 = 1.0

    bce_0 = F.binary_cross_entropy(
        torch.sigmoid(halt_logits[0]),
        torch.full_like(halt_logits[0], target_0),
    )
    bce_1 = F.binary_cross_entropy(
        torch.sigmoid(halt_logits[1]),
        torch.full_like(halt_logits[1], target_1),
    )
    expected = (bce_0 + bce_1) / 2

    assert torch.allclose(combined, expected, atol=1e-5), (
        f"两轮均值不符: combined={combined.item():.5f}, expected={expected.item():.5f}"
    )


def test_act_loss_wrong_length() -> None:
    """halt_logits 与 answer_qualities 长度不一致时应抛出 ValueError。"""
    fn = ACTLoss(lambda_step=0.1, gamma=0.9)
    with pytest.raises(ValueError):
        fn([_make_halt_logit(0.0)], answer_qualities=[0.5, 0.5])


# ---------------------------------------------------------------------------
# compute_nav_act_loss 测试
# ---------------------------------------------------------------------------


def _make_result(num_rounds: int = 1) -> dict:
    """构造最小化的 RecursiveRetriever result 字典（用于 compute_nav_act_loss 测试）。"""
    attn_weights_per_step = [
        _make_attn([0.7, 0.2, 0.1]),  # round0 L1
        _make_attn([0.6, 0.4]),  # round0 L2
        _make_attn([0.5, 0.3, 0.2]),  # round0 L3
    ]
    # 多轮时追加占位权重
    for _ in range(num_rounds - 1):
        attn_weights_per_step += [
            _make_attn([0.5, 0.3, 0.2]),
            _make_attn([0.6, 0.4]),
            _make_attn([0.4, 0.3, 0.3]),
        ]
    halt_logits = [_make_halt_logit(float(i)) for i in range(num_rounds)]
    return {
        "attn_weights_per_step": attn_weights_per_step,
        "halt_logits": halt_logits,
    }


def test_compute_nav_act_loss_keys() -> None:
    """返回字典应含 total / nav / act 三个键。"""
    result = _make_result()
    out = compute_nav_act_loss(
        result,
        gt_path=(0, 1, 2),
        answer_qualities=[0.7],
        nav_loss_fn=NavigationLoss(),
        act_loss_fn=ACTLoss(lambda_step=0.1, gamma=0.9),
        nav_weight=1.0,
        act_weight=0.1,
    )
    assert "total" in out, "缺少 'total' 键"
    assert "nav" in out, "缺少 'nav' 键"
    assert "act" in out, "缺少 'act' 键"


def test_compute_nav_act_loss_total() -> None:
    """total 应等于 nav_weight * nav + act_weight * act。"""
    nav_w, act_w = 1.0, 0.1
    result = _make_result()
    out = compute_nav_act_loss(
        result,
        gt_path=(0, 1, 2),
        answer_qualities=[0.7],
        nav_loss_fn=NavigationLoss(),
        act_loss_fn=ACTLoss(lambda_step=0.1, gamma=0.9),
        nav_weight=nav_w,
        act_weight=act_w,
    )
    expected_total = nav_w * out["nav"] + act_w * out["act"]
    assert torch.allclose(out["total"], expected_total, atol=1e-5), (
        f"total 不符: {out['total'].item():.5f} vs {expected_total.item():.5f}"
    )


def test_compute_nav_act_loss_uses_first_round_attn() -> None:
    """仅使用第一轮的 attn_weights_per_step[:3]，不受后续轮次影响。"""
    nav_fn = NavigationLoss()
    act_fn = ACTLoss(lambda_step=0.1, gamma=0.9)

    # 单轮 result
    result_1 = _make_result(num_rounds=1)
    # 多轮 result（第一轮 attn 相同，后续轮次不同）
    result_2 = _make_result(num_rounds=2)

    out_1 = compute_nav_act_loss(
        result_1,
        gt_path=(0, 1, 2),
        answer_qualities=[0.7],
        nav_loss_fn=nav_fn,
        act_loss_fn=act_fn,
        nav_weight=1.0,
        act_weight=0.0,  # act_weight=0 隔离 nav 部分
    )
    out_2 = compute_nav_act_loss(
        result_2,
        gt_path=(0, 1, 2),
        answer_qualities=[0.7, 0.8],
        nav_loss_fn=nav_fn,
        act_loss_fn=act_fn,
        nav_weight=1.0,
        act_weight=0.0,
    )
    # nav loss 应完全相同（第一轮 attn 一致）
    assert torch.allclose(out_1["nav"], out_2["nav"], atol=1e-5), (
        f"nav loss 应仅取第一轮: {out_1['nav'].item():.5f} vs {out_2['nav'].item():.5f}"
    )

"""
test_recursive_retriever.py — RecursiveRetriever 单元测试
=========================================================
覆盖各子组件（RMSNorm, SwiGLU, ReasoningBlock, ReasoningModule,
CrossAttentionSelector）以及主模型（RecursiveRetriever）的行为。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from video_tree_trm.config import RetrieverConfig
from video_tree_trm.recursive_retriever import (
    CrossAttentionSelector,
    ReasoningBlock,
    ReasoningModule,
    RecursiveRetriever,
    RMSNorm,
    SwiGLU,
)
from video_tree_trm.tree_index import IndexMeta, L1Node, L2Node, L3Node, TreeIndex


# ---------------------------------------------------------------------------
# 公共 Fixtures
# ---------------------------------------------------------------------------

D = 8  # 小 embed_dim，加速测试


def _make_config(max_rounds: int = 3) -> RetrieverConfig:
    """构造最小化 RetrieverConfig（embed_dim=8）。"""
    return RetrieverConfig(
        embed_dim=D,
        num_heads=2,
        L_layers=2,
        L_cycles=2,
        max_rounds=max_rounds,
        ffn_expansion=2.0,
        checkpoint=None,
    )


@pytest.fixture
def small_tree() -> TreeIndex:
    """2 L1 × 3 L2 × 4 L3，embed_dim=8 的小 TreeIndex。"""
    rng = np.random.default_rng(42)
    meta = IndexMeta(
        source_path="dummy.mp4",
        modality="video",
        embed_model="test",
        embed_dim=D,
    )

    def _l3(idx: int) -> L3Node:
        return L3Node(
            id=f"l3_{idx}",
            description=f"L3 节点 {idx}",
            embedding=rng.random(D).astype(np.float32),
        )

    def _l2(idx: int) -> L2Node:
        return L2Node(
            id=f"l2_{idx}",
            description=f"L2 节点 {idx}",
            embedding=rng.random(D).astype(np.float32),
            children=[_l3(idx * 10 + j) for j in range(4)],  # 4 个 L3 子节点
        )

    def _l1(idx: int) -> L1Node:
        return L1Node(
            id=f"l1_{idx}",
            summary=f"L1 节点 {idx}",
            embedding=rng.random(D).astype(np.float32),
            children=[_l2(idx * 10 + j) for j in range(3)],  # 3 个 L2 子节点
        )

    return TreeIndex(metadata=meta, roots=[_l1(0), _l1(1)])


# ---------------------------------------------------------------------------
# RMSNorm 测试
# ---------------------------------------------------------------------------


def test_rms_norm_normalizes() -> None:
    """输出形状不变，各样本归一化后的 RMS ≈ 1（weight 为全 1 时）。"""
    norm = RMSNorm(D)
    x = torch.randn(4, D)
    y = norm(x)

    assert y.shape == x.shape, "输出形状应与输入相同"

    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-5), f"RMS 应≈1，实际={rms}"


def test_rms_norm_weight_scale() -> None:
    """将 weight 全部设为 2，输出 RMS 应≈2。"""
    norm = RMSNorm(D)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(4, D)
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full((4,), 2.0), atol=1e-4), (
        f"缩放后 RMS 应≈2，实际={rms}"
    )


# ---------------------------------------------------------------------------
# SwiGLU 测试
# ---------------------------------------------------------------------------


def test_swiglu_shape() -> None:
    """输入 [B, D] → 输出 [B, D]，维度不变。"""
    ffn = SwiGLU(dim=D, hidden_dim=D * 2)
    x = torch.randn(3, D)
    y = ffn(x)
    assert y.shape == (3, D), f"SwiGLU 输出形状应为 (3, {D})，实际={y.shape}"


def test_swiglu_nonzero_output() -> None:
    """输出不应全零（线性层随机初始化 + SiLU 激活）。"""
    ffn = SwiGLU(dim=D, hidden_dim=D * 2)
    x = torch.randn(2, D)
    y = ffn(x)
    assert y.abs().sum().item() > 0, "SwiGLU 输出不应全零"


# ---------------------------------------------------------------------------
# ReasoningBlock 测试
# ---------------------------------------------------------------------------


def test_reasoning_block_residual() -> None:
    """输出形状一致，且与输入有非零差异（验证非恒等映射）。"""
    block = ReasoningBlock(dim=D, expansion=2.0)
    x = torch.randn(2, D)
    y = block(x)

    assert y.shape == x.shape, "ReasoningBlock 应保持形状不变"
    assert not torch.allclose(x, y), "ReasoningBlock 输出应与输入不同（非恒等）"


# ---------------------------------------------------------------------------
# ReasoningModule 测试
# ---------------------------------------------------------------------------


def test_reasoning_module_stacks() -> None:
    """L_layers=2 时，输出形状正确，injection 被有效融合（与 injection=0 有差异）。"""
    module = ReasoningModule(dim=D, L_layers=2, expansion=2.0)
    z = torch.randn(2, D)
    injection = torch.randn(2, D)

    out = module(z, injection)
    assert out.shape == (2, D), (
        f"ReasoningModule 输出形状应为 (2, {D})，实际={out.shape}"
    )

    zero_inj = module(z, torch.zeros(2, D))
    assert not torch.allclose(out, zero_inj), "injection 应影响输出"


# ---------------------------------------------------------------------------
# CrossAttentionSelector 测试
# ---------------------------------------------------------------------------


def test_cross_attention_selector_output_shapes() -> None:
    """selected_info [1,D]，attn_weights [1,N]，selected_idx [1]。"""
    sel = CrossAttentionSelector(embed_dim=D, num_heads=2)
    B, N = 1, 5
    state = torch.randn(B, D)
    candidates = torch.randn(B, N, D)

    selected_info, attn_weights, selected_idx = sel(state, candidates)

    assert selected_info.shape == (B, D), (
        f"selected_info 应为 ({B},{D})，实际={selected_info.shape}"
    )
    assert attn_weights.shape == (B, N), (
        f"attn_weights 应为 ({B},{N})，实际={attn_weights.shape}"
    )
    assert selected_idx.shape == (B,), (
        f"selected_idx 应为 ({B},)，实际={selected_idx.shape}"
    )


def test_cross_attention_selector_softmax_sum() -> None:
    """attn_weights 经 softmax，各行之和≈1。"""
    sel = CrossAttentionSelector(embed_dim=D, num_heads=2)
    B, N = 2, 4
    state = torch.randn(B, D)
    candidates = torch.randn(B, N, D)

    _, attn_weights, _ = sel(state, candidates)
    row_sums = attn_weights.sum(dim=-1)  # [B]
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
        f"softmax 行和应≈1，实际={row_sums}"
    )


def test_cross_attention_selector_single_candidate() -> None:
    """N=1 时 selected_idx 应为 0，attn_weights≈1.0。"""
    sel = CrossAttentionSelector(embed_dim=D, num_heads=2)
    state = torch.randn(1, D)
    candidates = torch.randn(1, 1, D)

    _, attn_weights, selected_idx = sel(state, candidates)
    assert selected_idx.item() == 0, (
        f"单候选时 selected_idx 应为 0，实际={selected_idx.item()}"
    )
    assert abs(attn_weights.item() - 1.0) < 1e-5, (
        f"单候选时 attn_weights 应≈1，实际={attn_weights.item()}"
    )


def test_cross_attention_selector_invalid_dim() -> None:
    """embed_dim 不能被 num_heads 整除时应抛出 AssertionError。"""
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        CrossAttentionSelector(embed_dim=9, num_heads=4)


# ---------------------------------------------------------------------------
# RecursiveRetriever 初始化测试
# ---------------------------------------------------------------------------


def test_retriever_init_bias() -> None:
    """q_head.bias 初始化应≈-5.0。"""
    retriever = RecursiveRetriever(_make_config())
    bias_val = retriever.q_head.bias.item()
    assert abs(bias_val - (-5.0)) < 1e-6, f"q_head.bias 应为 -5.0，实际={bias_val}"


# ---------------------------------------------------------------------------
# RecursiveRetriever.forward 测试
# ---------------------------------------------------------------------------


def test_retriever_forward_output_keys(small_tree: TreeIndex) -> None:
    """推理输出字典应含 paths / num_rounds / z_final 三个键。"""
    retriever = RecursiveRetriever(_make_config())
    retriever.eval()
    q = torch.randn(1, D)

    with torch.no_grad():
        result = retriever(q, small_tree)

    assert "paths" in result
    assert "num_rounds" in result
    assert "z_final" in result


def test_retriever_forward_internals(small_tree: TreeIndex) -> None:
    """return_internals=True 时输出应额外含 attn_weights_per_step 和 halt_logits。"""
    retriever = RecursiveRetriever(_make_config())
    retriever.eval()
    q = torch.randn(1, D)

    with torch.no_grad():
        result = retriever(q, small_tree, return_internals=True)

    assert "attn_weights_per_step" in result, "应含 attn_weights_per_step"
    assert "halt_logits" in result, "应含 halt_logits"
    assert len(result["attn_weights_per_step"]) == result["num_rounds"] * 3, (
        "每轮 3 步，共 num_rounds*3 项 attn_weights"
    )
    assert len(result["halt_logits"]) == result["num_rounds"], (
        "halt_logits 长度应等于 num_rounds"
    )


def test_retriever_training_runs_all_rounds(small_tree: TreeIndex) -> None:
    """训练模式下应始终跑满 max_rounds，不提前停止。"""
    max_rounds = 3
    retriever = RecursiveRetriever(_make_config(max_rounds=max_rounds))
    retriever.train()

    # 设 q_head 输出大正值，推理模式下本应触发 halt
    with torch.no_grad():
        retriever.q_head.bias.fill_(100.0)
        retriever.q_head.weight.fill_(0.0)

    q = torch.randn(1, D)
    result = retriever(q, small_tree)

    assert result["num_rounds"] == max_rounds, (
        f"训练模式下应跑满 {max_rounds} 轮，实际={result['num_rounds']}"
    )


def test_retriever_inference_halts(small_tree: TreeIndex) -> None:
    """推理模式下 halt_logit > 0 且 round_idx > 0 时应提前停止。"""
    max_rounds = 3
    retriever = RecursiveRetriever(_make_config(max_rounds=max_rounds))
    retriever.eval()

    # q_head 始终返回 100.0 → round_idx=1 时触发 halt
    with torch.no_grad():
        retriever.q_head.bias.fill_(100.0)
        retriever.q_head.weight.fill_(0.0)

    q = torch.randn(1, D)
    with torch.no_grad():
        result = retriever(q, small_tree)

    assert result["num_rounds"] < max_rounds, (
        f"推理 halt 应使 num_rounds < {max_rounds}，实际={result['num_rounds']}"
    )
    assert result["num_rounds"] == 2, (
        f"应在第 2 轮后停止（round_idx=1），实际={result['num_rounds']}"
    )


def test_retriever_z_shape(small_tree: TreeIndex) -> None:
    """z_final 应为形状 [1, D] 的 Tensor。"""
    retriever = RecursiveRetriever(_make_config())
    retriever.eval()
    q = torch.randn(1, D)

    with torch.no_grad():
        result = retriever(q, small_tree)

    z = result["z_final"]
    assert isinstance(z, torch.Tensor), "z_final 应为 Tensor"
    assert z.shape == (1, D), f"z_final 形状应为 (1, {D})，实际={z.shape}"


def test_retriever_path_indices_valid(small_tree: TreeIndex) -> None:
    """所有路径中 k1 < N1，k2 < N2，k3 < N3。"""
    N1 = len(small_tree.roots)
    retriever = RecursiveRetriever(_make_config())
    retriever.eval()
    q = torch.randn(1, D)

    with torch.no_grad():
        result = retriever(q, small_tree)

    for k1, k2, k3 in result["paths"]:
        N2 = len(small_tree.roots[k1].children)
        N3 = len(small_tree.roots[k1].children[k2].children)
        assert 0 <= k1 < N1, f"k1={k1} 越界 N1={N1}"
        assert 0 <= k2 < N2, f"k2={k2} 越界 N2={N2}"
        assert 0 <= k3 < N3, f"k3={k3} 越界 N3={N3}"

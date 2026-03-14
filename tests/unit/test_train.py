"""
test_train.py — train.py 单元测试
====================================
使用 unittest.mock.MagicMock + patch 隔离所有外部依赖（无真实 LLM / 文件 IO）。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from train import find_gt_path_text, find_gt_path_video, prepare_training_data, train
from video_tree_trm.tree_index import IndexMeta, L1Node, L2Node, L3Node, TreeIndex


# ---------------------------------------------------------------------------
# 辅助：构造测试用 TreeIndex
# ---------------------------------------------------------------------------

D = 8


def _make_meta(modality: str = "text") -> IndexMeta:
    return IndexMeta(
        source_path="dummy",
        modality=modality,
        embed_model="test",
        embed_dim=D,
    )


def _emb() -> np.ndarray:
    return np.zeros(D, dtype=np.float32)


def _make_text_tree() -> TreeIndex:
    """
    1 L1 × 2 L2 × 2 L3，L3 节点 raw_content 各不同。
    L3 (0,0,0): "apple orange"
    L3 (0,0,1): "banana grape"
    L3 (0,1,0): "cat dog"
    L3 (0,1,1): "elephant fish"
    """
    l3_00 = L3Node(id="l3_00", description="d", embedding=_emb(), raw_content="apple orange")
    l3_01 = L3Node(id="l3_01", description="d", embedding=_emb(), raw_content="banana grape")
    l3_10 = L3Node(id="l3_10", description="d", embedding=_emb(), raw_content="cat dog")
    l3_11 = L3Node(id="l3_11", description="d", embedding=_emb(), raw_content="elephant fish")
    l2_0 = L2Node(id="l2_0", description="d", embedding=_emb(), children=[l3_00, l3_01])
    l2_1 = L2Node(id="l2_1", description="d", embedding=_emb(), children=[l3_10, l3_11])
    l1 = L1Node(id="l1_0", summary="s", embedding=_emb(), children=[l2_0, l2_1])
    return TreeIndex(metadata=_make_meta("text"), roots=[l1])


def _make_video_tree() -> TreeIndex:
    """
    1 L1(0~30s) × 1 L2(5~20s) × 3 L3(timestamps: 6, 12, 18)
    """
    l3_0 = L3Node(id="l3_0", description="d", embedding=_emb(), timestamp=6.0)
    l3_1 = L3Node(id="l3_1", description="d", embedding=_emb(), timestamp=12.0)
    l3_2 = L3Node(id="l3_2", description="d", embedding=_emb(), timestamp=18.0)
    l2 = L2Node(
        id="l2_0", description="d", embedding=_emb(),
        time_range=(5.0, 20.0), children=[l3_0, l3_1, l3_2],
    )
    l1 = L1Node(
        id="l1_0", summary="s", embedding=_emb(),
        time_range=(0.0, 30.0), children=[l2],
    )
    return TreeIndex(metadata=_make_meta("video"), roots=[l1])


def _make_config(save_dir: str, dataset_path: str = "/tmp/data.jsonl") -> MagicMock:
    cfg = MagicMock()
    cfg.train.lr = 1e-3
    cfg.train.weight_decay = 0.0
    cfg.train.batch_size = 1
    cfg.train.max_epochs_phase1 = 2
    cfg.train.max_epochs_phase2 = 2
    cfg.train.nav_loss_weight = 1.0
    cfg.train.act_loss_weight = 0.5
    cfg.train.act_lambda_step = 0.1
    cfg.train.act_gamma = 0.9
    cfg.train.eval_interval = 1
    cfg.train.save_dir = save_dir
    cfg.train.dataset_path = dataset_path
    cfg.retriever.embed_dim = D
    cfg.retriever.num_heads = 2
    cfg.retriever.L_layers = 1
    cfg.retriever.L_cycles = 1
    cfg.retriever.max_rounds = 3
    cfg.retriever.ffn_expansion = 2.0
    cfg.retriever.checkpoint = None
    return cfg


# ---------------------------------------------------------------------------
# find_gt_path_text 测试
# ---------------------------------------------------------------------------


def test_find_gt_path_text_best_node() -> None:
    """返回与 answer 重叠度最高的节点路径（非全零打分时）。"""
    tree = _make_text_tree()
    # "apple" 与 (0,0,0) 的 "apple orange" 有完全匹配
    path = find_gt_path_text(tree, "apple")
    assert path == (0, 0, 0), f"预期 (0,0,0)，实际={path}"


def test_find_gt_path_text_no_overlap() -> None:
    """所有节点 F1=0 时，仍返回路径（第一个 L3 节点）而不是 None。"""
    tree = _make_text_tree()
    # "xyz" 与所有节点无重叠
    path = find_gt_path_text(tree, "xyz abc def")
    # 分数均为 0，best_score=-1 初始化，第一个节点 score=0 > -1，会返回非 None
    assert path is not None, "所有节点 F1=0 时仍应返回路径（最高分节点）"


def test_find_gt_path_text_empty_tree() -> None:
    """空 L1 列表时返回 None。"""
    tree = TreeIndex(metadata=_make_meta("text"), roots=[])
    path = find_gt_path_text(tree, "answer")
    assert path is None


# ---------------------------------------------------------------------------
# find_gt_path_video 测试
# ---------------------------------------------------------------------------


def test_find_gt_path_video_found() -> None:
    """timestamp 在 time_range 内，返回最近帧路径。"""
    tree = _make_video_tree()
    # timestamp=11 → 最近帧是 index=1 (ts=12)
    path = find_gt_path_video(tree, 11.0)
    assert path == (0, 0, 1), f"预期 (0,0,1)，实际={path}"


def test_find_gt_path_video_found_exact() -> None:
    """timestamp 恰好等于某帧时间戳，返回该帧。"""
    tree = _make_video_tree()
    path = find_gt_path_video(tree, 6.0)
    assert path == (0, 0, 0), f"预期 (0,0,0)，实际={path}"


def test_find_gt_path_video_not_found() -> None:
    """timestamp 超出所有 time_range，返回 None。"""
    tree = _make_video_tree()
    path = find_gt_path_video(tree, 100.0)
    assert path is None


def test_find_gt_path_video_none_time_range() -> None:
    """L1.time_range=None 的节点被跳过，返回 None。"""
    l3 = L3Node(id="l3_0", description="d", embedding=_emb(), timestamp=5.0)
    l2 = L2Node(id="l2_0", description="d", embedding=_emb(), time_range=None, children=[l3])
    l1 = L1Node(id="l1_0", summary="s", embedding=_emb(), time_range=None, children=[l2])
    tree = TreeIndex(metadata=_make_meta("video"), roots=[l1])
    path = find_gt_path_video(tree, 5.0)
    assert path is None, "time_range=None 的节点应被跳过"


# ---------------------------------------------------------------------------
# prepare_training_data 测试
# ---------------------------------------------------------------------------


def test_prepare_training_data_text(tmp_path: Path) -> None:
    """文本模式：正确解析 JSONL，返回含 query/tree/gt_path/answer 的 dict。"""
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps(
            {"query": "apple?", "answer": "apple", "source_path": "doc.txt", "modality": "text"}
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = _make_config(save_dir=str(tmp_path / "ckpt"), dataset_path=str(data_file))
    mock_tree = _make_text_tree()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.build_index.return_value = mock_tree

    with patch("train.Pipeline", return_value=mock_pipeline_instance):
        dataset = prepare_training_data(cfg)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["query"] == "apple?"
    assert sample["answer"] == "apple"
    assert sample["tree"] is mock_tree
    assert sample["gt_path"] is not None


def test_prepare_training_data_skips_none_gt(tmp_path: Path) -> None:
    """gt_path 推导为 None 的样本被跳过，不出现在返回列表中。"""
    data_file = tmp_path / "data.jsonl"
    # 视频样本但树为空，timestamp 找不到
    data_file.write_text(
        json.dumps(
            {
                "query": "what?", "answer": "ans",
                "source_path": "vid.mp4", "modality": "video", "timestamp": 999.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = _make_config(save_dir=str(tmp_path / "ckpt"), dataset_path=str(data_file))
    # 构建一个 timestamp 不匹配的空视频树
    empty_video_tree = TreeIndex(metadata=_make_meta("video"), roots=[])
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.build_index.return_value = empty_video_tree

    with patch("train.Pipeline", return_value=mock_pipeline_instance):
        dataset = prepare_training_data(cfg)

    assert len(dataset) == 0, "gt_path=None 的样本应被过滤"


# ---------------------------------------------------------------------------
# train() 测试
# ---------------------------------------------------------------------------


def _make_retriever_result(num_rounds: int = 2) -> dict:
    """构造 RecursiveRetriever.forward() 的 Mock 返回值。"""
    attn = [torch.softmax(torch.randn(1, 2), dim=-1)] * (num_rounds * 3)
    halt = [torch.tensor([[0.5]])] * num_rounds
    paths = [(0, 0, 0)] * num_rounds
    total = torch.tensor(0.5, requires_grad=True)
    return {
        "paths": paths,
        "num_rounds": num_rounds,
        "z_final": torch.zeros(1, D),
        "attn_weights_per_step": attn,
        "halt_logits": halt,
        "_total_for_test": total,
    }


def test_train_phase1_max_rounds_is_1(tmp_path: Path) -> None:
    """Phase 1 训练开始时 retriever.max_rounds 被设置为 1。"""
    cfg = _make_config(save_dir=str(tmp_path / "ckpt"))

    mock_retriever = MagicMock()
    mock_result = _make_retriever_result(num_rounds=1)
    # 让 backward 可调用
    mock_loss_dict = {"total": torch.tensor(0.1, requires_grad=True), "nav": torch.tensor(0.1), "act": torch.tensor(0.0)}
    mock_retriever.return_value = mock_result
    mock_retriever.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])

    observed_max_rounds: list = []

    def capture_max_rounds(*args, **kwargs):
        observed_max_rounds.append(mock_retriever.max_rounds)
        return mock_result

    mock_retriever.side_effect = capture_max_rounds

    dataset = [{"query": "q", "tree": _make_text_tree(), "gt_path": (0, 0, 0), "answer": "a"}]

    with patch("train.EmbeddingModel") as MockEmbed, \
         patch("train.RecursiveRetriever", return_value=mock_retriever), \
         patch("train.LLMClient"), \
         patch("train.compute_nav_act_loss", return_value=mock_loss_dict), \
         patch("train.prepare_training_data", return_value=dataset), \
         patch("train.torch.save"):

        MockEmbed.return_value.embed_tensor.return_value = torch.zeros(1, D)
        train(cfg)

    # Phase 1 期间捕获的 max_rounds 应为 1
    assert all(r == 1 for r in observed_max_rounds[:cfg.train.max_epochs_phase1]), (
        f"Phase 1 max_rounds 应为 1，观察到: {observed_max_rounds}"
    )


def test_train_phase2_restores_max_rounds(tmp_path: Path) -> None:
    """Phase 2 开始时 retriever.max_rounds 恢复为 config.retriever.max_rounds。"""
    cfg = _make_config(save_dir=str(tmp_path / "ckpt"))
    cfg.retriever.max_rounds = 3

    mock_retriever = MagicMock()
    mock_result = _make_retriever_result(num_rounds=2)
    mock_loss_dict = {"total": torch.tensor(0.1, requires_grad=True), "nav": torch.tensor(0.1), "act": torch.tensor(0.0)}
    mock_retriever.return_value = mock_result
    mock_retriever.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])

    observed_max_rounds: list = []
    call_count = [0]

    def capture(*args, **kwargs):
        call_count[0] += 1
        observed_max_rounds.append(mock_retriever.max_rounds)
        return mock_result

    mock_retriever.side_effect = capture

    dataset = [{"query": "q", "tree": _make_text_tree(), "gt_path": (0, 0, 0), "answer": "a"}]

    with patch("train.EmbeddingModel") as MockEmbed, \
         patch("train.RecursiveRetriever", return_value=mock_retriever), \
         patch("train.LLMClient") as MockLLM, \
         patch("train.compute_nav_act_loss", return_value=mock_loss_dict), \
         patch("train.prepare_training_data", return_value=dataset), \
         patch("train.torch.save"):

        MockEmbed.return_value.embed_tensor.return_value = torch.zeros(1, D)
        MockLLM.return_value.chat.return_value = "mock answer"
        train(cfg)

    # Phase 2 的 max_rounds 应为 config.retriever.max_rounds=3
    phase2_start = cfg.train.max_epochs_phase1  # Phase 1 共 2 epoch × 1 sample = 2 次调用
    if len(observed_max_rounds) > phase2_start:
        assert observed_max_rounds[phase2_start] == 3, (
            f"Phase 2 max_rounds 应为 3，观察到: {observed_max_rounds[phase2_start]}"
        )


def test_train_phase1_saves_checkpoint(tmp_path: Path) -> None:
    """eval_interval=1 时，每个 Phase 1 epoch 结束后调用 torch.save。"""
    cfg = _make_config(save_dir=str(tmp_path / "ckpt"))
    cfg.train.eval_interval = 1
    cfg.train.max_epochs_phase1 = 2
    cfg.train.max_epochs_phase2 = 0  # 不跑 Phase 2

    mock_retriever = MagicMock()
    mock_result = _make_retriever_result(num_rounds=1)
    mock_loss_dict = {"total": torch.tensor(0.1, requires_grad=True), "nav": torch.tensor(0.1), "act": torch.tensor(0.0)}
    mock_retriever.return_value = mock_result
    mock_retriever.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
    mock_retriever.state_dict.return_value = {}

    dataset = [{"query": "q", "tree": _make_text_tree(), "gt_path": (0, 0, 0), "answer": "a"}]

    with patch("train.EmbeddingModel") as MockEmbed, \
         patch("train.RecursiveRetriever", return_value=mock_retriever), \
         patch("train.LLMClient"), \
         patch("train.compute_nav_act_loss", return_value=mock_loss_dict), \
         patch("train.prepare_training_data", return_value=dataset), \
         patch("train.torch.save") as mock_save:

        MockEmbed.return_value.embed_tensor.return_value = torch.zeros(1, D)
        train(cfg)

    # Phase 1 共 2 epoch，每次都 save（eval_interval=1）
    assert mock_save.call_count >= 2, (
        f"torch.save 应被调用 ≥2 次，实际={mock_save.call_count}"
    )


def test_train_phase2_calls_llm_per_round(tmp_path: Path) -> None:
    """Phase 2 中 llm.chat 调用次数等于 num_rounds × epoch × sample_count。"""
    cfg = _make_config(save_dir=str(tmp_path / "ckpt"))
    cfg.train.max_epochs_phase1 = 0  # 跳过 Phase 1
    cfg.train.max_epochs_phase2 = 1

    num_rounds = 2
    mock_retriever = MagicMock()
    mock_result = _make_retriever_result(num_rounds=num_rounds)
    mock_loss_dict = {"total": torch.tensor(0.1, requires_grad=True), "nav": torch.tensor(0.1), "act": torch.tensor(0.0)}
    mock_retriever.return_value = mock_result
    mock_retriever.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])

    dataset = [{"query": "q", "tree": _make_text_tree(), "gt_path": (0, 0, 0), "answer": "a"}]

    mock_llm_instance = MagicMock()
    mock_llm_instance.chat.return_value = "mock answer"

    with patch("train.EmbeddingModel") as MockEmbed, \
         patch("train.RecursiveRetriever", return_value=mock_retriever), \
         patch("train.LLMClient", return_value=mock_llm_instance), \
         patch("train.compute_nav_act_loss", return_value=mock_loss_dict), \
         patch("train.prepare_training_data", return_value=dataset), \
         patch("train.torch.save"):

        MockEmbed.return_value.embed_tensor.return_value = torch.zeros(1, D)
        train(cfg)

    # 1 epoch × 1 sample × num_rounds = 2 次
    assert mock_llm_instance.chat.call_count == num_rounds, (
        f"llm.chat 应被调用 {num_rounds} 次，实际={mock_llm_instance.chat.call_count}"
    )

"""
训练管线
========
实现两阶段训练策略：

Phase 1 — 导航训练（单轮，max_rounds=1）
    目标: 训练 CrossAttentionSelector + ReasoningModule 正确导航到目标节点。
    损失: NavigationLoss（cross-entropy on attn_weights）。
    可训练: selector / L_level；冻结: text_embed / TreeIndex embeddings。

Phase 2 — ACT 训练（多轮，max_rounds=config.retriever.max_rounds）
    目标: 训练 q_head 判断何时停止检索。
    损失: NavigationLoss + act_loss_weight × ACTLoss。
    可训练: 全部（selector + L_level + q_head）。

数据集格式（JSONL）::

    # 文本模式
    {"query": "...", "answer": "...", "source_path": "path/to/doc.txt", "modality": "text"}
    # 视频模式
    {"query": "...", "answer": "...", "source_path": "path/to/video.mp4",
     "modality": "video", "timestamp": 12.5}

使用方式::

    from video_tree_trm.config import Config
    from video_tree_trm.train import train

    cfg = Config.load("config/default.yaml")
    train(cfg)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW

from utils.logger_system import ensure, log_msg
from video_tree_trm.answer_generator import token_f1
from video_tree_trm.config import Config
from video_tree_trm.embeddings import EmbeddingModel
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.losses import ACTLoss, NavigationLoss, compute_nav_act_loss
from video_tree_trm.pipeline import Pipeline
from video_tree_trm.recursive_retriever import RecursiveRetriever
from video_tree_trm.tree_index import TreeIndex


# ---------------------------------------------------------------------------
# GT 路径推导工具函数
# ---------------------------------------------------------------------------


def find_gt_path_text(
    tree: TreeIndex,
    answer: str,
) -> Optional[Tuple[int, int, int]]:
    """在文本树中，找到与 answer token F1 最高的 L3 节点路径。

    参数:
        tree:   已构建的 TreeIndex（modality="text"）。
        answer: 参考答案字符串，用于与 L3.raw_content 计算 F1。

    返回:
        F1 最高的节点路径 (l1_idx, l2_idx, l3_idx)，
        若树为空（无 L1 节点）则返回 None。

    实现细节:
        遍历全部 L3 节点，利用已有的 token_f1() 函数打分，
        返回得分最高的路径；多个节点得分相同时取遍历顺序中最先遇到的。
    """
    best_score: float = -1.0
    best_path: Optional[Tuple[int, int, int]] = None

    for i, l1 in enumerate(tree.roots):
        for j, l2 in enumerate(l1.children):
            for k, l3 in enumerate(l2.children):
                score = token_f1(l3.raw_content or "", answer)
                if score > best_score:
                    best_score = score
                    best_path = (i, j, k)

    return best_path


def find_gt_path_video(
    tree: TreeIndex,
    timestamp: float,
) -> Optional[Tuple[int, int, int]]:
    """在视频树中，找到最接近 timestamp 的 L3 帧节点路径。

    参数:
        tree:      已构建的 TreeIndex（modality="video"）。
        timestamp: 目标时间戳（秒）。

    返回:
        最接近 timestamp 的 L3 节点路径 (l1_idx, l2_idx, l3_idx)，
        若无任何节点的 time_range 包含 timestamp 则返回 None。

    实现细节:
        - time_range 为 None 的 L1/L2 节点直接跳过。
        - 在满足 time_range 包含 timestamp 的 L1/L2 下，
          找 L3.timestamp 与目标最近的节点；L3.timestamp 为 None 时
          该 L3 节点不参与比较。
    """
    for i, l1 in enumerate(tree.roots):
        if l1.time_range is None:
            continue
        if not (l1.time_range[0] <= timestamp <= l1.time_range[1]):
            continue
        for j, l2 in enumerate(l1.children):
            if l2.time_range is None:
                continue
            if not (l2.time_range[0] <= timestamp <= l2.time_range[1]):
                continue
            # 找最近 L3
            valid = [
                (k, l3) for k, l3 in enumerate(l2.children) if l3.timestamp is not None
            ]
            if not valid:
                continue
            k_star = min(valid, key=lambda kl: abs(kl[1].timestamp - timestamp))[0]
            return (i, j, k_star)

    return None


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def prepare_training_data(config: Config) -> List[Dict[str, Any]]:
    """离线预处理训练数据集。

    参数:
        config: 全局配置对象。

    返回:
        样本列表，每条为::

            {
                "query":    str,
                "tree":     TreeIndex,
                "gt_path":  Tuple[int, int, int],
                "answer":   str,
            }

        gt_path 无法推导（返回 None）的样本将被跳过。

    实现细节:
        - 从 config.train.dataset_path 读取 JSONL 文件，每行一个样本。
        - 使用 Pipeline.build_index 构建并缓存 TreeIndex。
        - 文本样本通过 find_gt_path_text 推导 gt_path；
          视频样本通过 find_gt_path_video 推导 gt_path（需 "timestamp" 字段）。
    """
    ensure(
        Path(config.train.dataset_path).is_file(),
        f"数据集文件不存在: {config.train.dataset_path}",
    )

    pipeline = Pipeline(config)
    dataset: List[Dict[str, Any]] = []

    with open(config.train.dataset_path, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    log_msg("INFO", "开始构建训练数据集", total_samples=len(raw_lines))

    for idx, line in enumerate(raw_lines):
        sample_raw = json.loads(line)
        query: str = sample_raw["query"]
        answer: str = sample_raw["answer"]
        source_path: str = sample_raw["source_path"]
        modality: str = sample_raw["modality"]

        # Phase 1: 构建树索引（带缓存）
        tree: TreeIndex = pipeline.build_index(source_path, modality)

        # Phase 2: 推导 GT 路径
        if modality == "text":
            gt_path = find_gt_path_text(tree, answer)
        else:
            timestamp: float = float(sample_raw["timestamp"])
            gt_path = find_gt_path_video(tree, timestamp)

        if gt_path is None:
            log_msg(
                "WARNING",
                "GT 路径推导失败，跳过该样本",
                sample_idx=idx,
                query=query[:50],
            )
            continue

        dataset.append(
            {"query": query, "tree": tree, "gt_path": gt_path, "answer": answer}
        )

    log_msg(
        "INFO",
        "数据集构建完成",
        total=len(raw_lines),
        kept=len(dataset),
    )
    return dataset


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """两阶段训练主循环。

    参数:
        config: 通过 ``Config.load()`` 加载的全局配置对象。

    实现细节:
        - Phase 1: 单轮导航训练，强制 max_rounds=1，仅使用 NavigationLoss。
        - Phase 2: 多轮 ACT 训练，恢复 max_rounds，加入 ACTLoss。
        - 每 eval_interval epoch 保存一次检查点到 save_dir。
        - 使用 AdamW 优化器。
    """
    # Phase 1: 公共初始化
    embed_model = EmbeddingModel(config.embed)
    retriever = RecursiveRetriever(config.retriever)

    nav_loss_fn = NavigationLoss()
    act_loss_fn = ACTLoss(
        lambda_step=config.train.act_lambda_step,
        gamma=config.train.act_gamma,
    )
    optimizer = AdamW(
        retriever.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    dataset = prepare_training_data(config)
    ensure(len(dataset) > 0, "训练数据集为空，请检查 dataset_path 和 GT 路径推导逻辑")

    save_dir = Path(config.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_msg(
        "INFO",
        "训练初始化完成",
        dataset_size=len(dataset),
        max_epochs_phase1=config.train.max_epochs_phase1,
        max_epochs_phase2=config.train.max_epochs_phase2,
    )

    # Phase 2: 导航训练（max_rounds=1）
    retriever.train()
    retriever.max_rounds = 1

    for epoch in range(config.train.max_epochs_phase1):
        epoch_loss = 0.0
        for sample in dataset:
            q = embed_model.embed_tensor(sample["query"])  # [1, D]

            result = retriever(q, sample["tree"], return_internals=True)
            loss_dict = compute_nav_act_loss(
                result=result,
                gt_path=sample["gt_path"],
                answer_qualities=[],
                nav_loss_fn=nav_loss_fn,
                act_loss_fn=act_loss_fn,
                nav_weight=config.train.nav_loss_weight,
                act_weight=0.0,  # Phase 1 不计算 ACT loss
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            epoch_loss += loss_dict["total"].item()

        avg_loss = epoch_loss / len(dataset)
        log_msg(
            "INFO",
            "Phase 1 epoch 完成",
            epoch=epoch + 1,
            total=config.train.max_epochs_phase1,
            avg_loss=round(avg_loss, 6),
        )

        if (epoch + 1) % config.train.eval_interval == 0:
            ckpt = save_dir / f"phase1_epoch{epoch + 1}.pt"
            torch.save(retriever.state_dict(), str(ckpt))
            log_msg("INFO", "Phase 1 检查点已保存", path=str(ckpt))

    # Phase 3: ACT 训练（全轮数）
    llm = LLMClient(config.llm)
    retriever.max_rounds = config.retriever.max_rounds

    for epoch in range(config.train.max_epochs_phase2):
        epoch_loss = 0.0
        for sample in dataset:
            q = embed_model.embed_tensor(sample["query"])
            result = retriever(q, sample["tree"], return_internals=True)

            # 每轮调用 LLM 计算答案质量
            qualities: List[float] = []
            for t in range(result["num_rounds"]):
                paths_so_far = result["paths"][: t + 1]
                nodes = [sample["tree"].get_node(*p) for p in paths_so_far]
                context = "\n".join(n.raw_content for n in nodes if n.raw_content)
                pred = llm.chat(f"上下文: {context}\n问题: {sample['query']}")
                qualities.append(token_f1(pred, sample["answer"]))

            loss_dict = compute_nav_act_loss(
                result=result,
                gt_path=sample["gt_path"],
                answer_qualities=qualities,
                nav_loss_fn=nav_loss_fn,
                act_loss_fn=act_loss_fn,
                nav_weight=config.train.nav_loss_weight,
                act_weight=config.train.act_loss_weight,
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            epoch_loss += loss_dict["total"].item()

        avg_loss = epoch_loss / len(dataset)
        log_msg(
            "INFO",
            "Phase 2 epoch 完成",
            epoch=epoch + 1,
            total=config.train.max_epochs_phase2,
            avg_loss=round(avg_loss, 6),
        )

        if (epoch + 1) % config.train.eval_interval == 0:
            ckpt = save_dir / f"phase2_epoch{epoch + 1}.pt"
            torch.save(retriever.state_dict(), str(ckpt))
            log_msg("INFO", "Phase 2 检查点已保存", path=str(ckpt))

    log_msg("INFO", "训练完成", save_dir=str(save_dir))

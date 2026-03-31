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

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW

import swanlab

from utils.logger_system import ensure, log_msg
from video_tree_trm.answer_generator import AnswerGenerator, token_f1
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
    correct_option: Optional[str] = None,
) -> Optional[Tuple[int, int, int]]:
    """在视频树中，找到 ground-truth L3 节点路径。

    优先根据 timestamp 定位；若 timestamp 为 0 且提供了 correct_option，
    则退化为在 L3 节点描述中寻找与正确选项相关度最高（Token F1）的节点。
    """
    # 1. 尝试根据时间戳定位
    if timestamp > 0:
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
                valid = [
                    (k, l3)
                    for k, l3 in enumerate(l2.children)
                    if l3.timestamp is not None
                ]
                if not valid:
                    continue
                k_star = min(valid, key=lambda kl: abs(kl[1].timestamp - timestamp))[0]
                return (i, j, k_star)

    # 2. 回退：基于描述文本与正确答案的匹配（适用于 VideoMME 这种无时间戳但有文本答案的情况）
    if correct_option:
        best_score = -1.0
        best_path = None
        for i, l1 in enumerate(tree.roots):
            for j, l2 in enumerate(l1.children):
                for k, l3 in enumerate(l2.children):
                    score = token_f1(l3.description or "", correct_option)
                    if score > best_score:
                        best_score = score
                        best_path = (i, j, k)
        if best_score > 0:
            return best_path

    return None


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def prepare_training_data(config: Config, embed_model: EmbeddingModel) -> List[Dict[str, Any]]:
    """离线预处理训练数据集。

    参数:
        config: 全局配置对象。
        embed_model: 嵌入模型（用于延迟嵌入树的节点）。

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
    tree_cache: Dict[str, TreeIndex] = {}  # 按 source_path 缓存已嵌入的树

    with open(config.train.dataset_path, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    log_msg("INFO", "开始构建训练数据集", total_samples=len(raw_lines))

    for idx, line in enumerate(raw_lines):
        sample_raw = json.loads(line)
        query: str = sample_raw["query"]
        answer: str = sample_raw["answer"]
        source_path: str = sample_raw["source_path"]
        modality: str = sample_raw["modality"]

        # 复用已嵌入的树，避免同一视频树的多个QA重复 embedding
        # build_index 会自动检查缓存、执行 embedding 并持久化到磁盘
        if source_path in tree_cache:
            tree = tree_cache[source_path]
        else:
            tree = pipeline.build_index(source_path, modality)
            # build_index 已自动执行 embedding 并保存到磁盘
            tree_cache[source_path] = tree

        # Phase 2: 推导 GT 路径
        if modality == "text":
            gt_path = find_gt_path_text(tree, answer)
        else:
            timestamp: float = float(sample_raw.get("timestamp", 0.0))
            # 对于 VideoMME，尝试提取正确选项的完整文本作为匹配依据
            correct_option_text = None
            if "options" in sample_raw and answer in "ABCD":
                idx = ord(answer) - ord("A")
                if 0 <= idx < len(sample_raw["options"]):
                    correct_option_text = sample_raw["options"][idx]
            
            gt_path = find_gt_path_video(tree, timestamp, correct_option_text)

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

    dataset = prepare_training_data(config, embed_model)
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

    # SwanLab 初始化
    swanlab.init(
        project="Video-Tree-TRM",
        experiment_name=f"phase1_nav_{config.train.max_epochs_phase1}ep",
        config={
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "embed_dim": config.retriever.embed_dim,
            "num_heads": config.retriever.num_heads,
            "max_epochs_phase1": config.train.max_epochs_phase1,
            "max_epochs_phase2": config.train.max_epochs_phase2,
            "dataset_size": len(dataset),
        },
    )

    # 加载验证集（如果存在）
    val_dataset: List[Dict[str, Any]] = []
    val_path = Path(config.train.dataset_path).parent / "val.jsonl"
    if val_path.is_file():
        # 临时修改 dataset_path 以加载验证集
        original_dataset_path = config.train.dataset_path
        config.train.dataset_path = str(val_path)
        val_dataset = prepare_training_data(config, embed_model)
        config.train.dataset_path = original_dataset_path
        log_msg("INFO", "验证集加载完成", val_samples=len(val_dataset))

    best_val_acc = 0.0  # 追踪最佳验证集准确率

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
        swanlab.log({"phase1/train_loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % config.train.eval_interval == 0:
            # 验证集评估
            if val_dataset:
                retriever.eval()
                correct_paths = 0
                correct_l1 = 0
                correct_l2 = 0
                correct_l3 = 0
                with torch.no_grad():
                    for sample in val_dataset:
                        q = embed_model.embed_tensor(sample["query"])
                        result = retriever(q, sample["tree"])
                        pred = result["paths"][0] if result["paths"] else None
                        gt_path = sample["gt_path"]

                        if pred is not None:
                            pk1, pk2, pk3 = pred.k1, pred.k2, pred.k3
                        else:
                            pk1, pk2, pk3 = -1, -1, -1

                        if pk1 == gt_path[0]:
                            correct_l1 += 1
                        if pk1 == gt_path[0] and pk2 == gt_path[1]:
                            correct_l2 += 1
                        if (pk1, pk2, pk3) == gt_path:
                            correct_paths += 1
                            correct_l3 += 1

                val_path_acc = correct_paths / len(val_dataset)
                val_l1_acc = correct_l1 / len(val_dataset)
                val_l2_acc = correct_l2 / len(val_dataset)
                val_l3_acc = correct_l3 / len(val_dataset)

                log_msg(
                    "INFO",
                    "Phase 1 验证集评估",
                    epoch=epoch + 1,
                    path_acc=round(val_path_acc, 4),
                    l1_acc=round(val_l1_acc, 4),
                    l2_acc=round(val_l2_acc, 4),
                    l3_acc=round(val_l3_acc, 4),
                )
                swanlab.log({
                    "phase1/val_path_acc": val_path_acc,
                    "phase1/val_l1_acc": val_l1_acc,
                    "phase1/val_l2_acc": val_l2_acc,
                    "phase1/val_l3_acc": val_l3_acc,
                    "epoch": epoch + 1,
                })

                # 保存最佳检查点
                if val_path_acc > best_val_acc:
                    best_val_acc = val_path_acc
                    best_ckpt = save_dir / "phase1_best.pt"
                    torch.save(retriever.state_dict(), str(best_ckpt))
                    log_msg("INFO", "Phase 1 最佳检查点已保存", path=str(best_ckpt), val_acc=round(best_val_acc, 4))

                retriever.train()

            # 常规检查点保存
            ckpt = save_dir / f"phase1_epoch{epoch + 1}.pt"
            torch.save(retriever.state_dict(), str(ckpt))
            log_msg("INFO", "Phase 1 检查点已保存", path=str(ckpt))

    # Phase 2: ACT 训练（多轮 + halt 决策）
    if config.train.max_epochs_phase2 > 0:
        # 加载 Phase 1 最佳检查点
        phase1_best = save_dir / "phase1_best.pt"
        if phase1_best.is_file():
            state = torch.load(str(phase1_best), map_location="cpu")
            retriever.load_state_dict(state)
            log_msg("INFO", "Phase 1 最佳检查点已加载", path=str(phase1_best))
        else:
            log_msg("WARNING", "Phase 1 最佳检查点不存在，使用当前权重继续")

        retriever.max_rounds = config.retriever.max_rounds
        retriever.train()
        log_msg("INFO", "Phase 2 ACT 训练开始", max_rounds=retriever.max_rounds)

        # SwanLab 重初始化
        swanlab.finish()
        swanlab.init(
            project="Video-Tree-TRM",
            experiment_name=f"phase2_act_{config.train.max_epochs_phase2}ep",
            config={
                "lr": config.train.lr,
                "max_rounds": config.retriever.max_rounds,
                "act_loss_weight": config.train.act_loss_weight,
                "max_epochs_phase2": config.train.max_epochs_phase2,
            },
        )

        best_val_acc_p2 = 0.0

        for epoch in range(config.train.max_epochs_phase2):
            epoch_loss = 0.0
            epoch_nav = 0.0
            epoch_act = 0.0

            for sample in dataset:
                q = embed_model.embed_tensor(sample["query"])
                result = retriever(q, sample["tree"], return_internals=True)

                # 用路径匹配度作为答案质量代理（避免每轮调用 LLM）
                qualities: List[float] = []
                for t in range(result["num_rounds"]):
                    pred = result["paths"][t] if t < len(result["paths"]) else None
                    if pred is not None:
                        gt = sample["gt_path"]
                        score = 0.0
                        if pred.k1 == gt[0]:
                            score = 0.33
                        if pred.k1 == gt[0] and pred.k2 == gt[1]:
                            score = 0.67
                        if (pred.k1, pred.k2, pred.k3) == gt:
                            score = 1.0
                        qualities.append(score)
                    else:
                        qualities.append(0.0)

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
                epoch_nav += loss_dict["nav"].item()
                epoch_act += loss_dict["act"].item()

            avg_loss = epoch_loss / len(dataset)
            avg_nav = epoch_nav / len(dataset)
            avg_act = epoch_act / len(dataset)
            log_msg(
                "INFO",
                "Phase 2 epoch 完成",
                epoch=epoch + 1,
                total=config.train.max_epochs_phase2,
                avg_loss=round(avg_loss, 6),
                avg_nav=round(avg_nav, 6),
                avg_act=round(avg_act, 6),
            )
            swanlab.log({
                "phase2/train_loss": avg_loss,
                "phase2/nav_loss": avg_nav,
                "phase2/act_loss": avg_act,
                "epoch": epoch + 1,
            })

            if (epoch + 1) % config.train.eval_interval == 0:
                if val_dataset:
                    retriever.eval()
                    correct_l1 = correct_l2 = correct_l3 = correct_paths = 0
                    with torch.no_grad():
                        for sample in val_dataset:
                            q = embed_model.embed_tensor(sample["query"])
                            result = retriever(q, sample["tree"])
                            pred = result["paths"][0] if result["paths"] else None
                            gt_path = sample["gt_path"]
                            if pred is not None:
                                pk1, pk2, pk3 = pred.k1, pred.k2, pred.k3
                            else:
                                pk1, pk2, pk3 = -1, -1, -1
                            if pk1 == gt_path[0]:
                                correct_l1 += 1
                            if pk1 == gt_path[0] and pk2 == gt_path[1]:
                                correct_l2 += 1
                            if (pk1, pk2, pk3) == gt_path:
                                correct_paths += 1
                                correct_l3 += 1

                    val_path_acc = correct_paths / len(val_dataset)
                    val_l1_acc = correct_l1 / len(val_dataset)
                    val_l2_acc = correct_l2 / len(val_dataset)
                    val_l3_acc = correct_l3 / len(val_dataset)

                    log_msg(
                        "INFO",
                        "Phase 2 验证集评估",
                        epoch=epoch + 1,
                        path_acc=round(val_path_acc, 4),
                        l1_acc=round(val_l1_acc, 4),
                        l2_acc=round(val_l2_acc, 4),
                        l3_acc=round(val_l3_acc, 4),
                    )
                    swanlab.log({
                        "phase2/val_path_acc": val_path_acc,
                        "phase2/val_l1_acc": val_l1_acc,
                        "phase2/val_l2_acc": val_l2_acc,
                        "phase2/val_l3_acc": val_l3_acc,
                        "epoch": epoch + 1,
                    })

                    if val_path_acc > best_val_acc_p2:
                        best_val_acc_p2 = val_path_acc
                        best_ckpt = save_dir / "phase2_best.pt"
                        torch.save(retriever.state_dict(), str(best_ckpt))
                        log_msg("INFO", "Phase 2 最佳检查点已保存", path=str(best_ckpt), val_acc=round(best_val_acc_p2, 4))

                    retriever.train()

                ckpt = save_dir / f"phase2_epoch{epoch + 1}.pt"
                torch.save(retriever.state_dict(), str(ckpt))
                log_msg("INFO", "Phase 2 检查点已保存", path=str(ckpt))

    swanlab.finish()
    log_msg("INFO", "训练完成", save_dir=str(save_dir))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_set_args(set_list: List[str]) -> Dict[str, Any]:
    """解析 --set key=value 参数为字典。"""
    result = {}
    for item in set_list:
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            # 尝试推断类型
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "null":
                value = None
            elif "." in value:
                try:
                    value = float(value)
                except ValueError:
                    pass
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-Tree-TRM 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="覆盖配置项，格式: key=value，如 --set train.dataset_path=data/xxx",
    )
    args = parser.parse_args()

    # 解析 --set 参数
    cli_overrides = parse_set_args(args.set)
    if cli_overrides:
        log_msg("INFO", "CLI 配置覆盖", overrides=cli_overrides)

    # 加载配置
    cfg = Config.load(args.config, cli_args=cli_overrides if cli_overrides else None)
    train(cfg)


if __name__ == "__main__":
    main()

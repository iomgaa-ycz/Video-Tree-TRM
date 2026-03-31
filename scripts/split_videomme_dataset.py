"""
VideoMME 数据集划分脚本
=======================
将已建树视频的 QA 数据划分为训练集 / 验证集 / 测试集。

支持两种模式：
  Stage 1: 从已建树 QA 中划分 80% 训练集（立即可用）
  Stage 2: 全部建树完成后，从剩余样本中划分 val/test

用法：
  # 阶段一：划分训练集
  python scripts/split_videomme_dataset.py --stage 1

  # 阶段二：划分验证集和测试集（需先完成全部建树）
  python scripts/split_videomme_dataset.py --stage 2
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/videomme/queries/sample_eval.jsonl"
DEFAULT_OUTPUT_DIR = "data/videomme/splits"
MANIFEST_NAME = "split_manifest.json"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def load_samples(input_path: str) -> tuple[list[dict], dict[str, list[int]]]:
    """加载 JSONL 文件，按 youtube_id 分组样本索引。

    返回:
        samples: 所有样本列表
        video_to_indices: youtube_id -> [sample_idx, ...]
    """
    samples = []
    video_to_indices = defaultdict(list)

    with open(input_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            samples.append(sample)
            yt_id = sample.get("youtube_id")
            if yt_id:
                video_to_indices[yt_id].append(idx)

    return samples, video_to_indices


def stratified_split_videos(
    video_to_indices: dict[str, list[int]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """按 youtube_id 分层采样，将视频分为训练集和剩余集。

    返回:
        train_ids: 训练集 youtube_id 集合
        remaining_ids: 剩余 youtube_id 集合
    """
    random.seed(seed)
    all_video_ids = list(video_to_indices.keys())
    random.shuffle(all_video_ids)

    n_train = max(1, int(len(all_video_ids) * train_ratio))
    train_ids = set(all_video_ids[:n_train])
    remaining_ids = set(all_video_ids[n_train:])

    return train_ids, remaining_ids


def remaining_split(
    video_to_indices: dict[str, list[int]],
    exclude_ids: set[str],
    val_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """从剩余视频中划分 val/test（均分）。

    返回:
        val_ids: 验证集 youtube_id 集合
        test_ids: 测试集 youtube_id 集合
    """
    random.seed(seed)
    remaining = [vid for vid in video_to_indices if vid not in exclude_ids]
    random.shuffle(remaining)

    n_val = max(1, len(remaining) // 2)
    val_ids = set(remaining[:n_val])
    test_ids = set(remaining[n_val:])

    return val_ids, test_ids


def save_split(
    samples: list[dict],
    video_to_indices: dict[str, list[int]],
    selected_ids: set[str],
    output_path: Path,
) -> int:
    """将选中 youtube_id 的样本写入 JSONL 文件，返回样本数量。"""
    selected_indices = set()
    for yt_id in selected_ids:
        selected_indices.update(video_to_indices[yt_id])

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in sorted(selected_indices):
            f.write(json.dumps(samples[idx], ensure_ascii=False) + "\n")

    return len(selected_indices)


def build_manifest(
    stage1: dict | None,
    stage2: dict | None,
    seed: int,
) -> dict:
    manifest = {"seed": seed}
    if stage1:
        manifest["stage1"] = stage1
    if stage2:
        manifest["stage2"] = stage2
    return manifest


def count_videos_in_samples(
    samples: list[dict],
    video_to_indices: dict[str, list[int]],
) -> dict[str, int]:
    """统计每个集合中的视频数量（通过 youtube_id）。"""
    result = {}
    for yt_id in video_to_indices:
        result[yt_id] = len(video_to_indices[yt_id])
    return result


# ---------------------------------------------------------------------------
# Stage 1: 划分训练集
# ---------------------------------------------------------------------------


def run_stage1(
    input_path: str,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """从已建树 QA 中划分训练集，输出 train.jsonl + manifest。"""
    print(f"[Stage 1] 加载样本: {input_path}")
    samples, video_to_indices = load_samples(input_path)

    total_samples = len(samples)
    total_videos = len(video_to_indices)
    print(f"[Stage 1] 共 {total_videos} 个视频，{total_samples} 条 QA 样本")

    # 按 youtube_id 分层采样
    train_ids, remaining_ids = stratified_split_videos(video_to_indices, train_ratio, seed)

    # 保存训练集
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    train_count = save_split(samples, video_to_indices, train_ids, train_path)

    # 生成 manifest
    manifest = build_manifest(
        stage1={
            "total_available": total_samples,
            "train_count": train_count,
            "train_videos": len(train_ids),
            "train_youtube_ids": sorted(train_ids),
            "remaining_count": total_samples - train_count,
            "remaining_videos": len(remaining_ids),
            "remaining_youtube_ids": sorted(remaining_ids),
            "path": str(train_path),
        },
        stage2=None,
        seed=seed,
    )

    manifest_path = output_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Stage 1] 完成!")
    print(f"  训练集: {train_count} 条 / {len(train_ids)} 个视频 -> {train_path}")
    print(f"  剩余: {total_samples - train_count} 条 / {len(remaining_ids)} 个视频 -> 待阶段二划分")
    print(f"  Manifest: {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# Stage 2: 划分验证集和测试集
# ---------------------------------------------------------------------------


def run_stage2(
    input_path: str,
    output_dir: Path,
    val_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    """从全部 QA 中排除训练集 youtube_id，剩余样本均分 val/test。"""
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest 文件不存在: {manifest_path}，请先运行 --stage 1")

    with open(manifest_path, encoding="utf-8") as f:
        old_manifest = json.load(f)

    stage1_info = old_manifest.get("stage1", {})
    train_ids = set(stage1_info.get("train_youtube_ids", []))

    print(f"[Stage 2] 加载样本: {input_path}")
    samples, video_to_indices = load_samples(input_path)

    total_samples = len(samples)
    total_videos = len(video_to_indices)
    print(f"[Stage 2] 共 {total_videos} 个视频，{total_samples} 条 QA 样本")

    # 统计训练集信息（从当前样本中）
    current_train_ids = train_ids & set(video_to_indices.keys())
    train_count = sum(len(video_to_indices[vid]) for vid in current_train_ids)
    print(f"[Stage 2] 训练集: {len(current_train_ids)} 个视频，{train_count} 条样本（来自 manifest）")

    # 从剩余视频中划分 val/test
    val_ids, test_ids = remaining_split(video_to_indices, train_ids, val_ratio, seed)

    # 保存 val.jsonl
    val_path = output_dir / "val.jsonl"
    val_count = save_split(samples, video_to_indices, val_ids, val_path)

    # 保存 test.jsonl
    test_path = output_dir / "test.jsonl"
    test_count = save_split(samples, video_to_indices, test_ids, test_path)

    # 更新 manifest
    new_manifest = build_manifest(
        stage1=stage1_info,
        stage2={
            "total_available": total_samples,
            "val_count": val_count,
            "val_videos": len(val_ids),
            "val_youtube_ids": sorted(val_ids),
            "val_path": str(val_path),
            "test_count": test_count,
            "test_videos": len(test_ids),
            "test_youtube_ids": sorted(test_ids),
            "test_path": str(test_path),
        },
        seed=seed,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)

    print(f"[Stage 2] 完成!")
    print(f"  验证集: {val_count} 条 / {len(val_ids)} 个视频 -> {val_path}")
    print(f"  测试集: {test_count} 条 / {len(test_ids)} 个视频 -> {test_path}")
    print(f"  Manifest 已更新: {manifest_path}")

    return new_manifest


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoMME 数据集划分")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"输入 JSONL 文件路径 (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="划分阶段: 1=划分训练集, 2=划分验证集和测试集",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例 (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.stage == 1:
        run_stage1(
            input_path=args.input,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    elif args.stage == 2:
        run_stage2(
            input_path=args.input,
            output_dir=output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

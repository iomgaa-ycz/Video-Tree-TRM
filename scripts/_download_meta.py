"""
_download_meta.py — VideoMME 元数据下载与长视频列表提取
==========================================================
从 HuggingFace `lmms-lab/Video-MME` 下载数据集元数据，
过滤 duration_category == "long"（30-60 分钟）的视频，
输出两个文件：
  - {meta_dir}/long_videos.jsonl    每行一条唯一长视频记录
  - {meta_dir}/long_videos_qa.jsonl 每行一条 QA 对（含 video_id）

使用方式（由 build_videomme_trees.sh 调用）:
    python _download_meta.py --meta-dir /data/videomme/metadata

依赖（在脚本中已安装）: datasets, huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载 VideoMME 元数据并提取长视频列表")
    p.add_argument("--meta-dir", required=True, help="元数据输出目录")
    p.add_argument(
        "--min-duration",
        type=int,
        default=1800,
        help="最短视频时长（秒），默认 1800（30 分钟）",
    )
    p.add_argument(
        "--max-duration",
        type=int,
        default=3600,
        help="最长视频时长（秒），默认 3600（60 分钟）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_dir = Path(args.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: 尝试加载 HuggingFace 数据集
    print("[meta] 正在从 HuggingFace 加载 lmms-lab/Video-MME ...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[meta][ERROR] 未安装 datasets 库，请先运行: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        # VideoMME 数据集，test split
        ds = load_dataset("lmms-lab/Video-MME", split="test")
    except Exception as e:
        print(f"[meta][ERROR] 数据集加载失败: {e}", file=sys.stderr)
        print("[meta] 请确认 HuggingFace 可访问，或配置 HF_ENDPOINT 镜像", file=sys.stderr)
        sys.exit(1)

    print(f"[meta] 数据集总条目数: {len(ds)}")

    # Phase 2: 过滤长视频
    # VideoMME 字段: video_id, youtube_id, url, duration, duration_category,
    #                domain, sub_category, question, answer, options
    seen_video_ids: set[str] = set()
    long_videos: list[dict] = []
    long_qa: list[dict] = []

    for row in ds:
        # 实际字段结构（lmms-lab/Video-MME 真实格式）:
        #   video_id      : "001", "002", ... （数据集内部序号）
        #   videoID       : YouTube 视频 ID（如 "fFjv93ACGo8"）
        #   url           : YouTube 完整链接
        #   duration      : "short" | "medium" | "long"（字符串类别，非秒数）
        #   domain        : 领域
        #   sub_category  : 细分类别
        #   question_id   : 问题序号
        #   question      : 问题文本
        #   options       : 选项列表（字符串）
        #   answer        : 正确选项字母
        duration_category = str(row.get("duration", "")).strip().lower()
        if duration_category != "long":
            continue

        youtube_id = row.get("videoID") or row.get("video_id", "")
        url = row.get("url", f"https://www.youtube.com/watch?v={youtube_id}")

        # 唯一视频记录（以 youtube_id 去重）
        if youtube_id not in seen_video_ids:
            seen_video_ids.add(youtube_id)
            long_videos.append(
                {
                    "video_id": row.get("video_id", ""),
                    "youtube_id": youtube_id,
                    "url": url,
                    "duration_category": duration_category,
                    "domain": row.get("domain", ""),
                    "sub_category": row.get("sub_category", ""),
                }
            )

        # QA 对记录
        long_qa.append(
            {
                "video_id": row.get("video_id", ""),
                "youtube_id": youtube_id,
                "question_id": row.get("question_id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "options": row.get("options", []),
                "duration_category": duration_category,
            }
        )

    # Phase 3: 写出文件
    videos_path = meta_dir / "long_videos.jsonl"
    qa_path = meta_dir / "long_videos_qa.jsonl"

    with open(videos_path, "w", encoding="utf-8") as f:
        for v in long_videos:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    with open(qa_path, "w", encoding="utf-8") as f:
        for q in long_qa:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"[meta] 长视频唯一数量: {len(long_videos)}")
    print(f"[meta] 长视频 QA 对数: {len(long_qa)}")
    print(f"[meta] 视频列表已保存: {videos_path}")
    print(f"[meta] QA 列表已保存:  {qa_path}")


if __name__ == "__main__":
    main()

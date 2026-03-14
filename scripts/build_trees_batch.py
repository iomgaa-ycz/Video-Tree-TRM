"""
批量视频建树脚本（视频间并行 + 视频内异步 L2/L3 并发）
==========================================================
扫描指定目录下所有 MP4，跳过已有 JSON 树的视频，
使用 ThreadPoolExecutor 进行视频间并行（--jobs），
每个视频内部使用 config.tree.concurrency 并发（默认 16）。

用法::

    # 串行（安全，适合 API 配额紧张时）
    conda run -n Video-Tree-TRM python scripts/build_trees_batch.py

    # 2 路视频并行（共 32 路 VLM 并发）
    conda run -n Video-Tree-TRM python scripts/build_trees_batch.py --jobs 2

    # 指定目录和配置
    conda run -n Video-Tree-TRM python scripts/build_trees_batch.py \\
        --video-dir data/videomme/videos \\
        --config config/videomme.yaml \\
        --jobs 2
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as cfwait
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_tree_trm.config import Config
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.video_tree_builder import VideoTreeBuilder
from utils.logger_system import log_msg


def _build_one(
    video_path: Path,
    cfg: Config,
) -> Tuple[str, bool, str]:
    """构建单个视频的 TreeIndex 并保存 JSON。

    参数:
        video_path: 视频文件绝对路径。
        cfg: 已加载的配置对象（由主进程共享，仅读取）。

    返回:
        (stem, success, message) 三元组。

    实现细节:
        每次调用独立初始化 LLMClient（避免多线程共享同一 httpx.Client 内部状态），
        使用 VideoTreeBuilder.build() 内部的异步事件循环（L2→L3→L1 链式并发）。
    """
    stem = video_path.stem
    try:
        # 每线程独立 LLMClient（httpx.Client 线程安全，但独立更稳健）
        vlm = LLMClient(cfg.vlm)
        builder = VideoTreeBuilder(vlm, cfg.tree)
        tree = builder.build(str(video_path))

        # 保存 JSON
        cache_dir = Path(cfg.tree.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{stem}_video.json"
        tree.save_json(str(out_path))

        l1 = len(tree.roots)
        l2 = sum(len(r.children) for r in tree.roots)
        l3 = sum(len(l2n.children) for r in tree.roots for l2n in r.children)
        msg = f"L1={l1} L2={l2} L3={l3} → {out_path}"
        log_msg("INFO", "视频建树完成", stem=stem, l1=l1, l2=l2, l3=l3)
        return stem, True, msg

    except Exception as e:  # noqa: BLE001
        log_msg("ERROR", "视频建树失败", stem=stem, error=str(e))
        return stem, False, str(e)


def main() -> None:
    """批量建树主函数：视频间 ThreadPoolExecutor 并行 + 视频内异步事件循环。"""
    parser = argparse.ArgumentParser(description="批量视频建树（视频间并行）")
    parser.add_argument(
        "--video-dir",
        default="data/videomme/videos",
        help="MP4 视频目录（默认: data/videomme/videos）",
    )
    parser.add_argument(
        "--config",
        default="config/videomme.yaml",
        help="配置文件路径（默认: config/videomme.yaml）",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="视频间并行数（默认: 1，每路视频内部已有 concurrency 路并发）",
    )
    args = parser.parse_args()

    # Phase 1: 加载配置（所有线程共享，只读）
    cfg = Config.load(args.config)
    log_msg(
        "INFO",
        "批量建树配置",
        video_dir=args.video_dir,
        cache_dir=cfg.tree.cache_dir,
        jobs=args.jobs,
        intra_concurrency=cfg.tree.concurrency,
    )

    # Phase 2: 扫描视频 + 过滤已有树
    video_dir = Path(args.video_dir)
    assert video_dir.is_dir(), f"视频目录不存在: {video_dir}"
    all_videos: List[Path] = sorted(video_dir.glob("*.mp4"))

    cache_dir = Path(cfg.tree.cache_dir)
    pending: List[Path] = [
        v for v in all_videos
        if not (cache_dir / f"{v.stem}_video.json").exists()
    ]
    skipped = len(all_videos) - len(pending)

    print(f"\n===== 批量建树 =====")
    print(f"  总视频数:    {len(all_videos)}")
    print(f"  已跳过(已建): {skipped}")
    print(f"  待处理:      {len(pending)}")
    print(f"  视频间并行:  {args.jobs}")
    print(f"  视频内并发:  {cfg.tree.concurrency}")
    print(f"  输出目录:    {cache_dir}\n")

    if not pending:
        print("所有视频均已建树，无需处理。")
        return

    # Phase 3: 异步视频间并行（ThreadPoolExecutor + FIRST_COMPLETED 事件循环）
    built: List[str] = []
    failed: List[Tuple[str, str]] = []
    start_total = time.time()

    pending_futures: Dict[Future, Path] = {}
    pending_queue = list(pending)  # 待提交队列
    pool = ThreadPoolExecutor(max_workers=args.jobs)

    # 初始填满 jobs 个任务
    while pending_queue and len(pending_futures) < args.jobs:
        video = pending_queue.pop(0)
        fut = pool.submit(_build_one, video, cfg)
        pending_futures[fut] = video
        print(f"[提交] {video.stem} ({len(built)+len(failed)+len(pending_futures)}/{len(pending)})")

    # 事件循环：完成一个，补一个
    done_count = 0
    while pending_futures:
        done, _ = cfwait(list(pending_futures), return_when=FIRST_COMPLETED)
        for fut in done:
            video = pending_futures.pop(fut)
            stem, success, msg = fut.result()
            done_count += 1
            elapsed = time.time() - start_total
            if success:
                built.append(stem)
                print(f"[OK {done_count}/{len(pending)}] {stem}  {msg}  (累计 {elapsed:.0f}s)")
            else:
                failed.append((stem, msg))
                print(f"[FAIL {done_count}/{len(pending)}] {stem}  {msg}")

            # 补充下一个任务（非阻塞提交）
            if pending_queue:
                next_video = pending_queue.pop(0)
                next_fut = pool.submit(_build_one, next_video, cfg)
                pending_futures[next_fut] = next_video
                print(f"[提交] {next_video.stem} ({done_count+len(pending_futures)}/{len(pending)})")

    pool.shutdown(wait=False)

    # Phase 4: 汇总
    total_elapsed = time.time() - start_total
    print(f"\n===== 汇总 =====")
    print(f"  成功: {len(built)}")
    print(f"  失败: {len(failed)}")
    print(f"  跳过: {skipped}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    if failed:
        print("\n  失败列表:")
        for stem, err in failed:
            print(f"    {stem}: {err}")

    log_msg(
        "INFO",
        "批量建树完成",
        built=len(built),
        failed=len(failed),
        skipped=skipped,
        elapsed_s=round(total_elapsed, 1),
    )


if __name__ == "__main__":
    main()

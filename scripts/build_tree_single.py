"""
单视频建树脚本（仅 VLM，不加载 EmbeddingModel）
================================================
直接调用 VideoTreeBuilder，跳过 Pipeline 的嵌入模型初始化。
结果保存为 JSON 到 cache/trees/ 目录。

用法::

    conda run -n Video-Tree-TRM python scripts/build_tree_single.py \
        --video data/videomme/videos/xKiRmesHWIA.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_tree_trm.config import Config
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.video_tree_builder import VideoTreeBuilder
from utils.logger_system import log_msg


def main() -> None:
    """构建单个视频的 TreeIndex，仅使用 VLM，不加载 EmbeddingModel。"""
    parser = argparse.ArgumentParser(description="单视频建树（仅 VLM）")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    args = parser.parse_args()

    # Phase 1: 加载配置 + 初始化 VLM
    cfg = Config.load(args.config)
    vlm = LLMClient(cfg.vlm)

    # Phase 2: 构建树（纯 VLM 描述，embedding=None）
    builder = VideoTreeBuilder(vlm, cfg.tree)
    tree = builder.build(args.video)

    # Phase 3: 保存 JSON
    stem = Path(args.video).stem
    cache_dir = Path(cfg.tree.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(cache_dir / f"{stem}_video.json")
    tree.save_json(out_path)

    log_msg("INFO", "建树完成，已保存", path=out_path)
    print(f"\n[完成] TreeIndex 已保存到: {out_path}")
    print(f"  L1 节点数: {len(tree.roots)}")
    total_l2 = sum(len(r.children) for r in tree.roots)
    total_l3 = sum(len(l2.children) for r in tree.roots for l2 in r.children)
    print(f"  L2 节点数: {total_l2}")
    print(f"  L3 节点数: {total_l3}")


if __name__ == "__main__":
    main()

"""
推理/演示入口
=============
从原始文档（文本或视频）构建 TreeIndex，执行问答。
与 ``train.py``（训练入口）配对，是 Video-Tree-TRM 的端到端演示入口。

子命令
------
- ``index``:  仅构建并缓存 TreeIndex，不执行问答。
- ``query``:  构建索引（或复用缓存）后执行问答。

使用示例::

    # 仅构建文本索引
    python main.py index --source data/doc.txt --modality text

    # 单次问答（视频）
    python main.py query --source data/video.mp4 --modality video \\
        --question "视频的主要内容是什么？"

    # 交互式多轮问答（文本）
    python main.py query --source data/doc.txt --modality text --interactive

    # 自定义配置
    python main.py query --source data/doc.txt --modality text \\
        --config config/default.yaml --env .env \\
        --question "文档结论是什么？"
"""

from __future__ import annotations

import argparse
import sys

from utils.logger_system import log_msg
from video_tree_trm.config import Config
from video_tree_trm.pipeline import Pipeline


# ---------------------------------------------------------------------------
# CLI 解析
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """解析 CLI 参数，返回 Namespace。

    返回:
        包含子命令及所有选项的 Namespace 对象。
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Video-Tree-TRM 推理/演示入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 公共参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="config/default.yaml",
        metavar="YAML",
        help="配置文件路径（默认: config/default.yaml）",
    )
    common.add_argument(
        "--env",
        default=".env",
        metavar="ENV",
        help="环境变量文件路径（默认: .env）",
    )
    common.add_argument(
        "--source",
        required=True,
        metavar="PATH",
        help="原始文件路径（文本文件或视频文件）",
    )
    common.add_argument(
        "--modality",
        required=True,
        choices=["text", "video"],
        help="文件模态：text 或 video",
    )

    subparsers = parser.add_subparsers(dest="command", title="子命令")
    subparsers.required = True

    # ── index 子命令 ──
    subparsers.add_parser(
        "index",
        parents=[common],
        help="构建并缓存 TreeIndex（不执行问答）",
        description="从原始文件构建三层树索引并保存到缓存目录。",
    )

    # ── query 子命令 ──
    query_parser = subparsers.add_parser(
        "query",
        parents=[common],
        help="构建索引并执行问答",
        description="构建索引后执行单次问答或交互式多轮问答。",
    )
    mode_group = query_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--question",
        metavar="TEXT",
        help="单次问答：问题字符串",
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="交互式多轮问答模式（输入 quit 退出）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> Config:
    """从 YAML + .env 加载配置。

    参数:
        args: CLI 解析结果，需含 config（YAML 路径）和 env（.env 路径）字段。

    返回:
        完整的 Config 实例。

    异常:
        FileNotFoundError: 配置文件不存在。
        TypeError: YAML 缺少必需字段。
    """
    log_msg("INFO", "加载配置", yaml_path=args.config, env_path=args.env)
    return Config.load(
        yaml_path=args.config,
        env_path=args.env,
    )


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    """index 子命令：构建并缓存 TreeIndex，打印缓存路径。

    参数:
        args: 含 source、modality、config、env 字段的 Namespace。
    """
    config = _load_config(args)
    pipeline = Pipeline(config)

    log_msg("INFO", "开始构建索引", source=args.source, modality=args.modality)
    tree = pipeline.build_index(args.source, args.modality)

    print(f"索引构建完成。模态={args.modality}，来源={args.source}")
    print(f"节点数量: L1={len(tree.roots)}")
    log_msg("INFO", "索引构建完成", l1_count=len(tree.roots))


def cmd_query(args: argparse.Namespace) -> None:
    """query 子命令：构建索引 → 单次或交互式问答。

    参数:
        args: 含 source、modality、question/interactive、config、env 字段的 Namespace。

    实现细节:
        - 先调用 build_index（缓存命中时直接加载，无额外开销）。
        - --question 模式：单次问答后退出。
        - --interactive 模式：循环读取用户输入，输入 'quit' 退出。
    """
    config = _load_config(args)
    pipeline = Pipeline(config)

    log_msg("INFO", "构建/加载索引", source=args.source, modality=args.modality)
    tree = pipeline.build_index(args.source, args.modality)

    if args.question:
        # Phase 1: 单次问答
        answer = pipeline.query(args.question, tree)
        print(f"\n问题: {args.question}")
        print(f"答案: {answer}\n")
        log_msg("INFO", "问答完成", question=args.question[:50])
    else:
        # Phase 2: 交互式多轮问答
        print("\n已进入交互式问答模式。输入 'quit' 退出。\n")
        while True:
            try:
                question = input("问题 (输入 'quit' 退出): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break

            if question.lower() in ("quit", "exit", "q"):
                print("再见。")
                break

            if not question:
                continue

            answer = pipeline.query(question, tree)
            print(f"答案: {answer}\n")
            log_msg("INFO", "交互式问答", question=question[:50])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """主入口：解析参数 → 分发子命令。

    退出码:
        0: 正常退出。
        1: 参数错误或运行时异常。
    """
    args = _parse_args()

    try:
        if args.command == "index":
            cmd_index(args)
        elif args.command == "query":
            cmd_query(args)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        log_msg("ERROR", f"main.py 异常退出: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

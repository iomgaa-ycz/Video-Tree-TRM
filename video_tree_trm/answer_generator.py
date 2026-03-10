"""
答案生成模块
============
根据 RecursiveRetriever 检索到的路径索引，从 TreeIndex 中提取节点内容，
按模态（文本 / 视频）组装 context，调用 LLM/VLM 生成最终答案。

同时提供 token_f1() 工具函数，供 Phase 2 训练循环评估每轮答案质量。

使用方式::

    from video_tree_trm.answer_generator import AnswerGenerator, token_f1
    from video_tree_trm.config import Config
    from video_tree_trm.llm_client import LLMClient

    cfg = Config.load("config/default.yaml")
    llm = LLMClient(cfg.llm)
    vlm = LLMClient(cfg.vlm)

    generator = AnswerGenerator(llm=llm, vlm=vlm)

    # 检索后生成答案（paths 来自 RecursiveRetriever.forward()["paths"]）
    answer = generator.generate(query="问题", paths=[(0, 1, 2)], tree=tree)

    # Phase 2 训练质量评估
    score = token_f1(prediction=answer, ground_truth="参考答案")
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from utils.logger_system import ensure, log_msg
from video_tree_trm.llm_client import LLMClient
from video_tree_trm.tree_index import TreeIndex

# ---------------------------------------------------------------------------
# Prompt 模板常量
# ---------------------------------------------------------------------------

_TEXT_PROMPT = "根据以下上下文回答问题。\n\n上下文:\n{context}\n\n问题: {query}"

_VIDEO_PROMPT = "根据以下关键帧回答问题。\n帧描述:\n{captions}\n\n问题: {query}"

_VIDEO_FALLBACK_PROMPT = (
    "根据以下视频片段描述回答问题（暂无关键帧图像）。\n"
    "片段描述:\n{captions}\n\n问题: {query}"
)


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------


class AnswerGenerator:
    """根据多轮检索路径组装 context，调用 LLM/VLM 生成最终答案。

    支持两种模态：
    - 文本模式: 提取 L3 节点的 raw_content，拼接后调用 LLM 生成答案。
    - 视频模式: 提取 L3 节点的 frame_path（帧图像）和 description（VLM 描述），
                调用 VLM 生成答案；若无帧路径则退化为仅用描述文本的 LLM 调用。

    属性:
        _llm: 文本对话客户端（文本模式 + 退化路径 + Phase 2 质量评估）。
        _vlm: 多模态客户端（视频模式，可为 None，但视频模式时必须提供）。
    """

    def __init__(self, llm: LLMClient, vlm: Optional[LLMClient] = None) -> None:
        """初始化 AnswerGenerator。

        参数:
            llm: LLMClient，文本对话客户端。文本模式及 Phase 2 训练均需此参数。
            vlm: LLMClient，多模态客户端。视频模式必填，文本模式可为 None。
        """
        self._llm = llm
        self._vlm = vlm
        log_msg(
            "INFO",
            "AnswerGenerator 初始化完成",
            has_vlm=vlm is not None,
        )

    def generate(
        self,
        query: str,
        paths: List[Tuple[int, int, int]],
        tree: TreeIndex,
    ) -> str:
        """根据检索路径生成最终答案。

        参数:
            query: 用户查询字符串。
            paths: 多轮检索路径列表，每项为 (k1, k2, k3) 整数三元组，
                   直接对应 RecursiveRetriever.forward()["paths"] 的输出。
            tree:  预构建的 TreeIndex，通过 get_node() 按路径访问 L3 节点。

        返回:
            生成的答案字符串。

        实现细节:
            - 通过 tree.metadata.modality 判断模态（"text" / "video"）。
            - 文本模式: 跳过 raw_content 为 None 的节点。
            - 视频模式: 若所有节点均无 frame_path，退化为 LLM 纯文本调用。
            - 视频模式使用 VLM 时须已提供 vlm 客户端。
        """
        ensure(len(paths) > 0, "paths 不能为空，至少需要一条检索路径")

        # 按路径提取 L3 节点
        nodes = [tree.get_node(k1, k2, k3) for k1, k2, k3 in paths]

        modality = tree.metadata.modality
        log_msg(
            "INFO",
            "开始答案生成",
            modality=modality,
            num_paths=len(paths),
            query=query[:50],
        )

        if modality == "text":
            return self._generate_text(query, nodes)
        else:
            return self._generate_video(query, nodes)

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _generate_text(self, query: str, nodes: list) -> str:
        """文本模式答案生成。

        参数:
            query: 用户查询。
            nodes: L3Node 列表。

        返回:
            LLM 生成的答案字符串。
        """
        context = "\n---\n".join(n.raw_content for n in nodes if n.raw_content)
        if not context:
            context = "（无可用上下文）"
        prompt = _TEXT_PROMPT.format(context=context, query=query)
        return self._llm.chat(prompt)

    def _generate_video(self, query: str, nodes: list) -> str:
        """视频模式答案生成。

        参数:
            query: 用户查询。
            nodes: L3Node 列表（含 frame_path / description）。

        返回:
            VLM/LLM 生成的答案字符串。

        实现细节:
            - 若存在有效帧路径：调用 VLM（多模态），需要 vlm 客户端。
            - 若所有帧路径为空：退化为 LLM 纯文本（用 description 作为上下文）。
        """
        frames = [n.frame_path for n in nodes if n.frame_path]
        captions = [n.description for n in nodes]
        caption_text = "\n".join(f"- {c}" for c in captions if c)

        if frames:
            ensure(
                self._vlm is not None,
                "视频模式需要 VLM 客户端，请在 AnswerGenerator.__init__() 中传入 vlm 参数",
            )
            prompt = _VIDEO_PROMPT.format(captions=caption_text, query=query)
            return self._vlm.chat_with_images(prompt, images=frames)  # type: ignore[union-attr]
        else:
            # 无帧路径：退化为 LLM 纯文本
            log_msg("INFO", "视频模式无可用帧，退化为 LLM 纯文本生成")
            prompt = _VIDEO_FALLBACK_PROMPT.format(captions=caption_text, query=query)
            return self._llm.chat(prompt)


# ---------------------------------------------------------------------------
# token_f1 工具函数
# ---------------------------------------------------------------------------


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token 级别 F1 分数，用于 Phase 2 训练质量评估。

    参考 SQuAD 评估方式，对 prediction 和 ground_truth 分词后
    计算 token 级别的 Precision / Recall / F1。

    参数:
        prediction:   模型生成的答案字符串。
        ground_truth: 标注的参考答案字符串。

    返回:
        F1 分数，范围 [0.0, 1.0]。prediction 或 ground_truth 为空时返回 0.0。

    实现细节:
        - 统一转小写、按空格分词（适用于英文/中英混合场景）。
        - common = Counter(pred) & Counter(gt)，支持重复词的正确统计。
    """
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

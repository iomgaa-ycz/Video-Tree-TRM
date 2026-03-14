"""
LLM/VLM 客户端模块
==================
统一封装 LLM（纯文本）和 VLM（多模态）API 调用。

仅支持 OpenAI-compatible 接口，通过配置 api_url + model 适配不同服务商
（如 Qwen DashScope、OpenAI、本地推理服务等）。

使用方式::

    from video_tree_trm.llm_client import LLMClient
    from video_tree_trm.config import Config

    cfg = Config.load("config/default.yaml")
    llm = LLMClient(cfg.llm)
    answer = llm.chat("你好，请介绍一下自己。")

    vlm = LLMClient(cfg.vlm)
    answer = vlm.chat_with_images("图中有什么？", images=["frame.jpg"])
"""

from __future__ import annotations

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import httpx
import openai

from utils.logger_system import log_exception, log_msg
from video_tree_trm.config import LLMConfig, VLMConfig

# 502/503 时的重试参数
_RETRY_STATUS_CODES = {502, 503}
_MAX_RETRIES = 20          # 最多重试次数（约等待 20+ 分钟）
_RETRY_BASE_WAIT = 60      # 首次等待 60 秒
_RETRY_MAX_WAIT = 300      # 单次等待上限 5 分钟


def _call_with_retry(fn, label: str):
    """对 fn() 调用执行指数退避重试（仅重试 502/503）。

    参数:
        fn: 无参调用的函数，返回 API response。
        label: 日志标识（如方法名）。

    返回:
        fn() 的返回值。

    异常:
        openai.OpenAIError: 超过最大重试次数或非可重试错误时抛出。
    """
    wait = _RETRY_BASE_WAIT
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except openai.InternalServerError as exc:
            status = getattr(exc, "status_code", None)
            if status not in _RETRY_STATUS_CODES:
                raise
            log_msg(
                "WARNING",
                f"{label} 遇到 {status}，等待 {wait}s 后重试",
                attempt=attempt,
                max_retries=_MAX_RETRIES,
            )
            time.sleep(wait)
            wait = min(wait * 2, _RETRY_MAX_WAIT)
    raise RuntimeError(f"{label} 已重试 {_MAX_RETRIES} 次仍失败")


class LLMClient:
    """OpenAI-compatible LLM/VLM 统一客户端。

    支持纯文本对话、多模态对话（VLM）和批量并发对话。
    通过 config.api_url 和 config.model 切换不同服务商，无需修改代码。

    属性:
        _config: LLMConfig 或 VLMConfig 配置对象。
        _client: openai.OpenAI 客户端实例。
    """

    def __init__(self, config: Union[LLMConfig, VLMConfig]) -> None:
        """初始化 LLM/VLM 客户端。

        参数:
            config: LLMConfig 或 VLMConfig，包含 api_key、api_url、model 等参数。

        异常:
            ValueError: api_key 或 api_url 为空时抛出。
        """
        if not config.api_key:
            raise ValueError(
                "LLMClient 初始化失败: config.api_key 不能为空，请在 .env 中设置"
            )
        if not config.api_url:
            raise ValueError(
                "LLMClient 初始化失败: config.api_url 不能为空，请在 config/default.yaml 中设置"
            )

        self._config = config
        # 显式绕过系统代理（http_proxy/https_proxy），直连 GPUStack 内网地址
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_url,
            http_client=httpx.Client(proxy=None),
        )
        log_msg(
            "INFO", "LLMClient 初始化完成", model=config.model, api_url=config.api_url
        )

    def chat(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """纯文本单轮对话。

        参数:
            prompt: 用户输入文本。
            max_tokens: 最大生成 token 数，为 None 时使用 config.max_tokens。

        返回:
            生成的文本字符串。

        异常:
            openai.OpenAIError: API 调用失败时抛出（记录日志后重新抛出）。
        """
        messages = self._build_messages(prompt)
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        try:
            response = _call_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=self._config.temperature,
                ),
                label="LLMClient.chat",
            )
            return response.choices[0].message.content
        except Exception as exc:
            log_exception("LLMClient.chat 调用失败", exc)
            raise

    def chat_with_images(
        self,
        prompt: str,
        images: List[str],
        max_tokens: Optional[int] = None,
    ) -> str:
        """多模态单轮对话（VLM）。

        参数:
            prompt: 文本指令。
            images: 图像列表，每项可为本地文件路径或已编码的 base64 字符串
                    （格式 "data:image/jpeg;base64,..."）。
            max_tokens: 最大生成 token 数，为 None 时使用 config.max_tokens。

        返回:
            生成的文本字符串。

        异常:
            openai.OpenAIError: API 调用失败时抛出（记录日志后重新抛出）。
        """
        encoded = [self._encode_image(img) for img in images]
        messages = self._build_messages(prompt, images=encoded)
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        try:
            response = _call_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=self._config.temperature,
                ),
                label="LLMClient.chat_with_images",
            )
            return response.choices[0].message.content
        except Exception as exc:
            log_exception("LLMClient.chat_with_images 调用失败", exc)
            raise

    def batch_chat(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """批量纯文本并发对话，保序返回。

        使用 ThreadPoolExecutor（max_workers=8）并发调用 chat()，
        结果按输入顺序排列。

        参数:
            prompts: 文本输入列表。
            max_tokens: 最大生成 token 数，为 None 时使用 config.max_tokens。

        返回:
            与 prompts 等长的生成文本列表，顺序与输入对应。

        异常:
            Exception: 任意单条请求失败时，异常从对应的 Future 中传播。
        """
        results: List[str] = [""] * len(prompts)

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务，保留索引用于保序
            future_to_idx = {
                executor.submit(self.chat, prompt, max_tokens): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    # ── 私有辅助方法 ──────────────────────────────────────────────────────────

    def _encode_image(self, path_or_b64: str) -> str:
        """将图像转换为 data URI 格式的 base64 字符串。

        参数:
            path_or_b64: 本地文件路径，或已是 "data:image/...;base64,..." 格式的字符串。

        返回:
            "data:image/jpeg;base64,<base64数据>" 格式字符串。

        异常:
            FileNotFoundError: 指定路径文件不存在时抛出。
        """
        # 已是 data URI，直接返回
        if "base64," in path_or_b64:
            return path_or_b64

        # 本地文件路径，读取并编码
        if not os.path.exists(path_or_b64):
            raise FileNotFoundError(f"图像文件不存在: {path_or_b64}")

        with open(path_or_b64, "rb") as f:
            raw = f.read()

        b64_data = base64.b64encode(raw).decode("utf-8")
        # 根据扩展名判断 MIME 类型
        ext = os.path.splitext(path_or_b64)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64_data}"

    def _build_messages(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> List[Dict]:
        """拼装 OpenAI-compatible 消息结构。

        参数:
            prompt: 文本指令。
            images: 已编码的 base64 data URI 列表（可为 None）。

        返回:
            OpenAI messages 格式的列表。

        实现细节:
            - 无图像：content 为纯字符串。
            - 有图像：content 为列表，图像在前，文本在后。
        """
        if not images:
            return [{"role": "user", "content": prompt}]

        content: List[Dict] = [
            {"type": "image_url", "image_url": {"url": img}} for img in images
        ]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

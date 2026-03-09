"""
LLMClient 单元测试
==================
测试覆盖:
  - TestLLMChatMock: mock OpenAI 客户端的纯文本对话功能
  - TestVLMChatMock: mock 环境下的多模态图像对话功能
  - TestConfigValidation: 配置校验（api_key/api_url 缺失）
  - TestRealLLMChat: 真实 API 集成测试（需 .env 配置）

运行::

    conda run -n Video-Tree-TRM python -m pytest tests/unit/test_llm_client.py -v \\
      --cov=video_tree_trm/llm_client --cov-report=term-missing
"""

from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from video_tree_trm.config import LLMConfig, VLMConfig
from video_tree_trm.llm_client import LLMClient


# ---------------------------------------------------------------------------
# 辅助：构造最小配置对象（避免加载真实 YAML）
# ---------------------------------------------------------------------------


def _make_llm_config(
    api_key: str = "sk-test",
    api_url: str = "https://api.example.com/v1",
    model: str = "test-model",
    max_tokens: int = 128,
    temperature: float = 0.1,
) -> LLMConfig:
    """构造测试用 LLMConfig，所有字段可覆盖。"""
    return LLMConfig(
        backend="openai",
        api_key=api_key,
        api_url=api_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _make_vlm_config(
    api_key: str = "sk-test",
    api_url: str = "https://api.example.com/v1",
    model: str = "test-vlm",
    max_tokens: int = 128,
    temperature: float = 0.1,
) -> VLMConfig:
    """构造测试用 VLMConfig。"""
    return VLMConfig(
        backend="openai",
        api_key=api_key,
        api_url=api_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _mock_completion(content: str) -> MagicMock:
    """构造 openai.ChatCompletion 返回值的 Mock。"""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# TestLLMChatMock — 纯文本对话（Mock）
# ---------------------------------------------------------------------------


class TestLLMChatMock:
    """使用 mock openai.OpenAI 测试 chat() 和 batch_chat()。"""

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_returns_string(self, mock_openai_cls: MagicMock) -> None:
        """chat() 应返回 API 返回内容的字符串。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("你好！")

        llm = LLMClient(_make_llm_config())
        result = llm.chat("你好")

        assert result == "你好！"
        assert isinstance(result, str)

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_uses_config_max_tokens(self, mock_openai_cls: MagicMock) -> None:
        """未传 max_tokens 时，应使用 config.max_tokens 的值。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("ok")

        cfg = _make_llm_config(max_tokens=256)
        llm = LLMClient(cfg)
        llm.chat("test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_overrides_max_tokens(self, mock_openai_cls: MagicMock) -> None:
        """显式传入 max_tokens 时，应覆盖 config.max_tokens。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("ok")

        cfg = _make_llm_config(max_tokens=256)
        llm = LLMClient(cfg)
        llm.chat("test", max_tokens=64)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 64

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_batch_chat_order_preserved(self, mock_openai_cls: MagicMock) -> None:
        """batch_chat() 应按输入顺序返回结果，即使并发完成顺序不同。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # 每次调用返回不同内容
        responses = ["结果0", "结果1", "结果2"]
        mock_client.chat.completions.create.side_effect = [
            _mock_completion(r) for r in responses
        ]

        llm = LLMClient(_make_llm_config())
        results = llm.batch_chat(["prompt0", "prompt1", "prompt2"])

        assert len(results) == 3
        assert results == responses

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_batch_chat_empty_list(self, mock_openai_cls: MagicMock) -> None:
        """batch_chat() 传入空列表时，应返回空列表。"""
        mock_openai_cls.return_value = MagicMock()
        llm = LLMClient(_make_llm_config())
        assert llm.batch_chat([]) == []


# ---------------------------------------------------------------------------
# TestVLMChatMock — 多模态对话（Mock）
# ---------------------------------------------------------------------------


class TestVLMChatMock:
    """使用 mock openai.OpenAI 测试 chat_with_images()。"""

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_with_images_encodes_local_path(self, mock_openai_cls: MagicMock) -> None:
        """传入本地文件路径时，消息中应包含 data:image/jpeg;base64, 前缀。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("图中有猫")

        # 创建临时 JPEG 文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # 最小 JPEG header
            tmp_path = f.name

        try:
            vlm = LLMClient(_make_vlm_config())
            result = vlm.chat_with_images("图中有什么？", images=[tmp_path])

            assert result == "图中有猫"
            # 验证消息结构包含 base64 编码图像
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            content = call_kwargs["messages"][0]["content"]
            assert isinstance(content, list)
            image_items = [c for c in content if c.get("type") == "image_url"]
            assert len(image_items) == 1
            assert image_items[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp_path)

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_with_images_accepts_b64(self, mock_openai_cls: MagicMock) -> None:
        """传入已有 base64 字符串时，不应重复编码，直接透传。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("ok")

        b64_str = "data:image/jpeg;base64," + base64.b64encode(b"fake").decode()
        vlm = LLMClient(_make_vlm_config())
        vlm.chat_with_images("描述图片", images=[b64_str])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_items = [c for c in content if c.get("type") == "image_url"]
        assert image_items[0]["image_url"]["url"] == b64_str

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_with_images_png_mime(self, mock_openai_cls: MagicMock) -> None:
        """PNG 文件应编码为 data:image/png;base64, 格式。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("ok")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            tmp_path = f.name

        try:
            vlm = LLMClient(_make_vlm_config())
            vlm.chat_with_images("描述图片", images=[tmp_path])

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            content = call_kwargs["messages"][0]["content"]
            image_items = [c for c in content if c.get("type") == "image_url"]
            assert image_items[0]["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp_path)

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_chat_with_images_message_structure(self, mock_openai_cls: MagicMock) -> None:
        """多模态消息中图像应在文本之前。"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_completion("ok")

        b64_str = "data:image/jpeg;base64," + base64.b64encode(b"img").decode()
        vlm = LLMClient(_make_vlm_config())
        vlm.chat_with_images("提问", images=[b64_str])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        # 最后一项为 text
        assert content[-1]["type"] == "text"
        assert content[-1]["text"] == "提问"
        # 前面各项为 image_url
        for item in content[:-1]:
            assert item["type"] == "image_url"

    def test_encode_image_file_not_found(self) -> None:
        """_encode_image 传入不存在的路径时，应抛出 FileNotFoundError。"""
        with patch("video_tree_trm.llm_client.openai.OpenAI"):
            llm = LLMClient(_make_llm_config())

        with pytest.raises(FileNotFoundError):
            llm._encode_image("/nonexistent/path/image.jpg")


# ---------------------------------------------------------------------------
# TestConfigValidation — 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """测试 LLMClient 初始化时的配置校验逻辑。"""

    def test_missing_api_key_raises(self) -> None:
        """api_key 为空时应抛出 ValueError。"""
        cfg = _make_llm_config(api_key="")
        with pytest.raises(ValueError, match="api_key"):
            LLMClient(cfg)

    def test_missing_api_url_raises(self) -> None:
        """api_url 为空时应抛出 ValueError。"""
        cfg = _make_llm_config(api_url="")
        with pytest.raises(ValueError, match="api_url"):
            LLMClient(cfg)

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_valid_config_initializes(self, mock_openai_cls: MagicMock) -> None:
        """有效配置应正常初始化，不抛出异常。"""
        mock_openai_cls.return_value = MagicMock()
        cfg = _make_llm_config()
        client = LLMClient(cfg)
        assert client is not None

    @patch("video_tree_trm.llm_client.openai.OpenAI")
    def test_vlm_config_accepted(self, mock_openai_cls: MagicMock) -> None:
        """VLMConfig 也应被正常接受。"""
        mock_openai_cls.return_value = MagicMock()
        cfg = _make_vlm_config()
        client = LLMClient(cfg)
        assert client is not None


# ---------------------------------------------------------------------------
# TestRealLLMChat — 真实 API 集成测试（需 .env）
# ---------------------------------------------------------------------------


class TestRealLLMChat:
    """调用真实 LLM API 进行集成测试。

    需要 .env 中配置有效的 LLM_API_KEY / LLM_API_URL / LLM_MODEL。
    """

    def test_real_chat(self, real_config) -> None:  # noqa: ANN001
        """真实 API 单轮对话，应返回非空字符串。"""
        llm = LLMClient(real_config.llm)
        result = llm.chat("请用一句话回答：天空是什么颜色？", max_tokens=32)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_real_batch_chat(self, real_config) -> None:  # noqa: ANN001
        """真实 API 批量对话，应返回与输入等长的非空字符串列表。"""
        llm = LLMClient(real_config.llm)
        prompts = ["1+1等于几？请只回答数字。", "2+2等于几？请只回答数字。"]
        results = llm.batch_chat(prompts, max_tokens=16)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

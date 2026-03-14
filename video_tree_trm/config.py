"""
配置管理模块
============
定义所有超参数的 dataclass 类型（无默认值）+ 多源加载。

三层优先级: CLI args > .env > YAML，统一归口到 Config dataclass。

使用方式::

    from video_tree_trm.config import Config

    cfg = Config.load("config/default.yaml")
    cfg = Config.load("config/default.yaml", cli_args={"retriever.num_heads": "8"})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import dotenv_values

from utils.logger_system import ensure, log_msg


# ---------------------------------------------------------------------------
# 子配置 Dataclass（全部无默认值，YAML 必须写全）
# ---------------------------------------------------------------------------


@dataclass
class TreeConfig:
    """树索引构建参数。

    属性:
        max_paragraphs_per_l2: 每个 L2 节点包含的最大段落数（文本模式）。
        l1_segment_duration: L1 段时长，秒（视频模式）。
        l2_clip_duration: L2 clip 时长，秒（视频模式）。
        l3_fps: L3 帧提取频率（视频模式）。
        l2_representative_frames: L2 VLM 描述用的代表帧数。
        cache_dir: TreeIndex 缓存目录。
    """

    max_paragraphs_per_l2: int
    l1_segment_duration: float
    l2_clip_duration: float
    l3_fps: float
    l2_representative_frames: int
    cache_dir: str
    concurrency: int


@dataclass
class EmbedConfig:
    """嵌入模型参数。

    属性:
        backend: 嵌入后端类型，"local"（sentence-transformers）或 "remote"（OpenAI 兼容 API）。
        model_name: 本地模式为 HuggingFace 模型名，远程模式为 API 模型名。
        embed_dim: 嵌入维度 D。
        device: 推理设备，"cuda" 或 "cpu"（仅本地模式使用）。
        api_key: 远程模式 API 密钥，从 .env 加载。本地模式为空串。
        api_url: 远程模式 API 端点。本地模式为空串。
    """

    backend: str
    model_name: str
    embed_dim: int
    device: str
    api_key: str
    api_url: str


@dataclass
class LLMConfig:
    """大语言模型参数。

    属性:
        backend: 后端类型，"qwen" | "openai" | "ollama"。
        api_key: API 密钥，从 .env 加载。
        model: 模型名称。
        api_url: API 端点 URL。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度。
    """

    backend: str
    api_key: str
    model: str
    api_url: str
    max_tokens: int
    temperature: float


@dataclass
class VLMConfig:
    """视觉语言模型参数。

    属性:
        backend: 后端类型，"qwen" | "openai" | "ollama"。
        api_key: API 密钥，从 .env 加载。
        model: 模型名称。
        api_url: API 端点 URL。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度。
    """

    backend: str
    api_key: str
    model: str
    api_url: str
    max_tokens: int
    temperature: float


@dataclass
class RetrieverConfig:
    """TRM 检索器参数。

    属性:
        embed_dim: 嵌入维度，须与 EmbedConfig.embed_dim 一致。
        num_heads: Cross-Attention 头数。
        L_layers: ReasoningModule 层数。
        L_cycles: 每级推理迭代次数。
        max_rounds: ACT 最大遍历轮次。
        ffn_expansion: SwiGLU 扩展比。
        checkpoint: 训练好的模型权重路径，推理时必填。
    """

    embed_dim: int
    num_heads: int
    L_layers: int
    L_cycles: int
    max_rounds: int
    ffn_expansion: float
    checkpoint: Optional[str]


@dataclass
class TrainConfig:
    """训练参数。

    属性:
        lr: 学习率。
        weight_decay: 权重衰减。
        batch_size: 批大小。
        max_epochs_phase1: Phase 1 导航训练轮数。
        max_epochs_phase2: Phase 2 ACT 训练轮数。
        nav_loss_weight: 导航损失权重。
        act_loss_weight: ACT 损失权重。
        act_lambda_step: ACT 步数惩罚系数。
        act_gamma: ACT 折扣因子。
        eval_interval: 每 N epoch 评估一次。
        save_dir: 模型权重保存目录。
        dataset: 数据集名称，"longbench" | "narrativeqa" | "videomme"。
        dataset_path: 数据集路径。
    """

    lr: float
    weight_decay: float
    batch_size: int
    max_epochs_phase1: int
    max_epochs_phase2: int
    nav_loss_weight: float
    act_loss_weight: float
    act_lambda_step: float
    act_gamma: float
    eval_interval: int
    save_dir: str
    dataset: str
    dataset_path: str


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

_SECTION_TO_CLASS: dict[str, type] = {
    "tree": TreeConfig,
    "embed": EmbedConfig,
    "llm": LLMConfig,
    "vlm": VLMConfig,
    "retriever": RetrieverConfig,
    "train": TrainConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，override 优先覆盖 base。

    参数:
        base: 基础字典。
        override: 覆盖字典。

    返回:
        合并后的新字典。
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dotpath(d: dict, key: str, value: Any) -> None:
    """通过点路径设置嵌套字典的值。

    支持 "retriever.num_heads" 风格的路径，自动拆分并逐级写入。

    参数:
        d: 目标字典。
        key: 点分隔的路径，如 "retriever.num_heads"。
        value: 要设置的值。
    """
    parts = key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _coerce_value(raw: str, target_type: type) -> Any:
    """将 CLI 字符串值转换为目标类型。

    参数:
        raw: 原始字符串值。
        target_type: 目标 Python 类型。

    返回:
        转换后的值。
    """
    if target_type is bool:
        return raw.lower() in ("true", "1", "yes")
    if target_type is type(None):
        return None if raw.lower() in ("none", "null", "") else raw
    return target_type(raw)


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """全局配置容器。

    统一管理所有子模块配置，通过 ``Config.load()`` 加载。

    属性:
        tree: 树索引构建参数。
        embed: 嵌入模型参数。
        llm: 大语言模型参数。
        vlm: 视觉语言模型参数。
        retriever: TRM 检索器参数。
        train: 训练参数。
    """

    tree: TreeConfig
    embed: EmbedConfig
    llm: LLMConfig
    vlm: VLMConfig
    retriever: RetrieverConfig
    train: TrainConfig

    @classmethod
    def load(
        cls,
        yaml_path: str,
        cli_args: Optional[dict[str, str]] = None,
        env_path: Optional[str] = None,
    ) -> "Config":
        """三层合并加载配置。

        优先级: CLI args > .env > YAML。

        参数:
            yaml_path: YAML 配置文件路径。
            cli_args: CLI 覆盖参数，键为点路径（如 "retriever.num_heads"），值为字符串。
            env_path: .env 文件路径，默认为项目根目录的 .env。

        返回:
            完整的 Config 实例。

        异常:
            FileNotFoundError: YAML 文件不存在。
            TypeError: YAML 中缺少必需字段。
        """
        # Phase 1: 读取 YAML
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_file, encoding="utf-8") as f:
            base_dict: dict = yaml.safe_load(f)

        # Phase 2: 读取 .env 覆盖敏感字段
        if env_path is None:
            env_file = Path(yaml_path).parent.parent / ".env"
        else:
            env_file = Path(env_path)

        if env_file.exists():
            env_vars = dotenv_values(str(env_file))
            env_overrides: dict[str, dict[str, str]] = {}
            # .env 变量名 → (配置节, 字段名) 的映射
            _ENV_MAP: dict[str, tuple[str, str]] = {
                "LLM_API_KEY": ("llm", "api_key"),
                "LLM_MODEL": ("llm", "model"),
                "LLM_API_URL": ("llm", "api_url"),
                "VLM_API_KEY": ("vlm", "api_key"),
                "VLM_MODEL": ("vlm", "model"),
                "VLM_API_URL": ("vlm", "api_url"),
                "EMBED_BACKEND": ("embed", "backend"),
                "EMBED_MODEL": ("embed", "model_name"),
                "EMBED_API_KEY": ("embed", "api_key"),
                "EMBED_API_URL": ("embed", "api_url"),
            }
            for env_name, (section, field) in _ENV_MAP.items():
                if env_vars.get(env_name):
                    env_overrides.setdefault(section, {})[field] = env_vars[env_name]
            base_dict = _deep_merge(base_dict, env_overrides)

        # Phase 3: CLI args 覆盖
        if cli_args:
            for dotpath, value in cli_args.items():
                _apply_dotpath(base_dict, dotpath, value)

        # Phase 4: 构造 dataclass（缺字段自动抛 TypeError）
        sections = {}
        for section_name, dc_class in _SECTION_TO_CLASS.items():
            section_data = base_dict.get(section_name, {})
            if not isinstance(section_data, dict):
                raise TypeError(
                    f"配置节 '{section_name}' 必须是字典，实际为 {type(section_data)}"
                )
            sections[section_name] = dc_class(**section_data)

        config = cls(**sections)

        # 校验: embed_dim 一致性
        ensure(
            config.embed.embed_dim == config.retriever.embed_dim,
            f"embed.embed_dim ({config.embed.embed_dim}) 与 "
            f"retriever.embed_dim ({config.retriever.embed_dim}) 不一致",
        )

        log_msg("INFO", "配置加载完成", yaml=yaml_path)
        return config

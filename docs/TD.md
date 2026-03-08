# 技术方案（TD）— Video-Tree-TRM

## 技术决策

- 单机本地执行，无服务端/数据库，所有数据 pickle/JSON 序列化到本地文件。
- 节点选择使用 **Cross-Attention**（学习 W_q/W_k/W_v/W_o 投影），替代简单 cosine 路由，更强表达力。
- L_level 推理模块使用 **MLP-based**（RMSNorm + SwiGLU），因操作对象为单向量 `[B, D]`，非序列，无需 self-attention。
- 三个可学习组件（CrossAttentionSelector, ReasoningModule, q_head）**跨层级共享权重**，与 TRM 原设计一致。
- 文本嵌入器（text_embed）**冻结不训练**，TreeIndex 中所有 embedding 为预计算静态值。
- 训练分两阶段：Phase 1 纯导航监督（单轮），Phase 2 加入 ACT halt（多轮）。
- MVP 优先文本模态（LongBench），视频模态（VideoMME）后续扩展。
- 配置管理：dataclass（无默认值，纯类型定义）+ YAML（全量配置）+ .env（敏感信息），优先级 CLI args > .env > YAML，三者统一归口到 dataclass。

---

## 目录

- [技术决策](#技术决策)
- [模块设计](#1-模块设计)
  - [tree_index.py](#11-tree_indexpy--统一数据结构)
  - [embeddings.py](#12-embeddingspy--嵌入服务)
  - [llm_client.py](#13-llm_clientpy--llmvlm-客户端)
  - [text_tree_builder.py](#14-text_tree_builderpy--文本树构建)
  - [video_tree_builder.py](#15-video_tree_builderpy--视频树构建)
  - [recursive_retriever.py](#16-recursive_retrieverpy--trm-递归检索器)
  - [losses.py](#17-lossespy--损失函数)
  - [answer_generator.py](#18-answer_generatorpy--答案生成)
  - [pipeline.py](#19-pipelinepy--端到端管线)
  - [config.py](#110-configpy--配置管理)
- [训练管线](#2-训练管线)
- [实验计划](#3-实验计划)
- [文件结构与依赖](#4-文件结构与依赖)

---

## 1. 模块设计

### 1.1 tree_index.py — 统一数据结构

**文件**: `video_tree_trm/tree_index.py`
**职责**: 定义三层树索引的节点类型、序列化/反序列化、嵌入矩阵提取。

```python
@dataclass
class IndexMeta:
    source_path: str           # 原始文件路径
    modality: str              # "text" | "video"
    embed_model: str           # 嵌入器名称, e.g. "BAAI/bge-base-zh-v1.5"
    embed_dim: int             # 嵌入维度 D
    created_at: str            # ISO 时间戳

@dataclass
class L3Node:
    id: str
    description: str           # 视频=VLM帧描述, 文本=原始段落
    embedding: ndarray         # [D], text_embed(description)
    raw_content: Optional[str] # 原始文本（文本模式）
    frame_path: Optional[str]  # 帧图像路径（视频模式）
    timestamp: Optional[float] # 帧时间戳（视频模式）

@dataclass
class L2Node:
    id: str
    description: str           # 1-2句片段描述
    embedding: ndarray         # [D]
    time_range: Optional[Tuple[float, float]]
    children: List[L3Node]

@dataclass
class L1Node:
    id: str
    summary: str               # 2-3句聚合摘要
    embedding: ndarray         # [D]
    time_range: Optional[Tuple[float, float]]
    children: List[L2Node]

@dataclass
class TreeIndex:
    metadata: IndexMeta
    roots: List[L1Node]
```

**关键方法**:

```python
class TreeIndex:
    def l1_embeddings(self) -> ndarray:
        """返回所有 L1 嵌入矩阵 [N1, D]"""

    def l2_embeddings_of(self, l1_idx: int) -> ndarray:
        """返回指定 L1 下所有 L2 子节点嵌入 [N2, D]"""

    def l3_embeddings_of(self, l1_idx: int, l2_idx: int) -> ndarray:
        """返回指定 L2 下所有 L3 子节点嵌入 [N3, D]"""

    def get_node(self, l1: int, l2: int, l3: int) -> L3Node:
        """按路径索引获取 L3 节点"""

    def save(self, path: str) -> None:
        """pickle 序列化到文件"""

    @classmethod
    def load(cls, path: str) -> "TreeIndex":
        """从文件反序列化"""
```

**依赖**: numpy, pickle（标准库）

---

### 1.2 embeddings.py — 嵌入服务

**文件**: `video_tree_trm/embeddings.py`
**职责**: 封装文本嵌入器，支持本地 sentence-transformers 和远程 OpenAI 兼容 API 双后端，冻结不训练。

```python
class EmbeddingModel:
    """文本嵌入器封装（冻结），支持本地/远程双后端。"""

    def __init__(self, config: EmbedConfig):
        """
        根据 config.backend 初始化:
          - "local": 加载 sentence-transformers 模型，冻结参数
          - "remote": 初始化 OpenAI 兼容 API 客户端
        """

    @property
    def dim(self) -> int:
        """嵌入维度 D"""

    def embed(self, texts: Union[str, List[str]]) -> ndarray:
        """
        文本 → 嵌入向量 (L2 归一化)
        Args:
            texts: 单条或批量文本
        Returns:
            [N, D] ndarray（单条时 N=1，每行 L2 范数为 1.0）
        """

    def embed_tensor(self, texts: Union[str, List[str]]) -> Tensor:
        """同 embed()，返回 torch.Tensor [N, D]（float32）"""

    # 内部方法
    def _embed_local(self, texts: List[str]) -> ndarray:
        """sentence-transformers 本地推理，torch.no_grad() + normalize_embeddings=True"""

    def _embed_remote(self, texts: List[str]) -> ndarray:
        """OpenAI 兼容 API: client.embeddings.create() → 提取向量 → L2 归一化"""
```

**远程模式示例** (GPUStack qwen3-embedding):
```python
# .env
EMBED_API_KEY=sk-xxx
EMBED_API_URL=http://gpu-host:8080/v1

# config/default.yaml
embed:
  backend: "remote"
  model_name: "qwen3-embedding-4b"
  embed_dim: 2048
  device: "cpu"       # 远程模式不使用
  api_key: ""         # 从 .env 覆盖
  api_url: ""         # 从 .env 覆盖
```

**依赖**: sentence-transformers（本地模式）, openai SDK（远程模式）, torch, numpy

---

### 1.3 llm_client.py — LLM/VLM 客户端

**文件**: `video_tree_trm/llm_client.py`
**职责**: 统一封装 LLM（纯文本）和 VLM（多模态）API 调用，支持多后端。

```python
class LLMClient:
    """LLM/VLM API 统一客户端"""

    def __init__(self, backend: str, api_key: str, model: str, **kwargs):
        """
        Args:
            backend: "qwen" | "openai" | "ollama"
            api_key: API 密钥（从 .env 读取）
            model: 模型名称
        """

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        """纯文本对话，返回生成文本"""

    def chat_with_images(
        self, prompt: str, images: List[str], max_tokens: int = 256
    ) -> str:
        """
        多模态对话（VLM）
        Args:
            prompt: 文本指令
            images: 图像路径列表或 base64 列表
        Returns:
            生成文本
        """

    def batch_chat(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """批量文本对话（并发或顺序）"""
```

**依赖**: openai SDK（兼容 Qwen/OpenAI/Ollama 接口）, python-dotenv

---

### 1.4 text_tree_builder.py — 文本树构建

**文件**: `video_tree_trm/text_tree_builder.py`
**职责**: 长文本 → TreeIndex，实现 L2 轴心构建策略。

```python
class TextTreeBuilder:
    """文本模态树构建器"""

    def __init__(self, embed_model: EmbeddingModel, llm: LLMClient, config: TreeConfig):
        self.embed = embed_model
        self.llm = llm
        self.config = config

    def build(self, text: str, source_path: str) -> TreeIndex:
        """
        完整构建流程:
          Step 1: 结构切分 → L1/L2 边界
          Step 2: L2 先行 → LLM 摘要 + 嵌入
          Step 3: L3 向下 → 原始段落 + 嵌入
          Step 4: L1 向上 → 聚合 L2 摘要 + 嵌入
          Step 5: 组装 TreeIndex
        """

    # ── 内部方法 ──

    def _segment_text(self, text: str) -> List[List[str]]:
        """
        Step 1: 结构切分
        Returns:
            sections[i] = [paragraph_1, paragraph_2, ...]
            外层 = L1 段, 内层段落组 = L2 单元
        策略:
            有 ToC → 正则解析章节标题
            无 ToC → LLM 单次调用语义分段
        """

    def _build_l2(self, paragraphs: List[str]) -> L2Node:
        """
        Step 2: 段落组 → L2 节点
        prompt: "用1-2句话描述以下段落的核心内容，与同级小节形成区分: {text}"
        """

    def _build_l3_from_paragraphs(self, paragraphs: List[str]) -> List[L3Node]:
        """
        Step 3: 每个段落 → L3 节点
        description = raw_content = 原始段落文本（不做摘要）
        embedding = text_embed(段落文本)
        """

    def _build_l1(self, l2_children: List[L2Node]) -> L1Node:
        """
        Step 4: 聚合 L2 描述 → L1 摘要
        prompt: "总结以下小节(2-3句): {l2_descriptions}"
        """
```

**依赖**: tree_index, embeddings, llm_client

---

### 1.5 video_tree_builder.py — 视频树构建

**文件**: `video_tree_trm/video_tree_builder.py`
**职责**: 长视频 → TreeIndex，实现 L2 轴心构建策略 + VLM 帧描述。

```python
class VideoTreeBuilder:
    """视频模态树构建器"""

    def __init__(self, embed_model: EmbeddingModel, vlm: LLMClient, config: TreeConfig):
        self.embed = embed_model
        self.vlm = vlm
        self.config = config

    def build(self, video_path: str) -> TreeIndex:
        """
        完整构建流程:
          Step 1: 时间切分 → L1/L2 时间边界
          Step 2: L2 先行 → 代表帧 VLM 描述 + 嵌入
          Step 3: L3 向下 → 注入 L2 上下文 VLM 帧描述 + 嵌入
          Step 4: L1 向上 → 聚合 L2 描述 + 嵌入
          Step 5: 组装 TreeIndex
        """

    # ── 内部方法 ──

    def _segment_video(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Step 1: 视频 → L1 时间区间列表
        策略: 固定步长 l1_segment_duration 或场景检测
        """

    def _extract_frames(self, video_path: str, time_range: Tuple[float, float], fps: float) -> List[Tuple[str, float]]:
        """
        提取指定时间范围内的帧
        Returns: [(frame_path, timestamp), ...]
        """

    def _build_l2_video(self, video_path: str, clip_range: Tuple[float, float]) -> L2Node:
        """
        Step 2: 每个 L2 clip → 采样 2-3 张代表帧 → VLM 描述 (1-2句)
        """

    def _build_l3_video(self, frames: List[Tuple[str, float]], l2_description: str) -> List[L3Node]:
        """
        Step 3: 注入 L2 上下文的 VLM 批量帧描述
        prompt = f'''
        该片段的整体内容: "{l2_description}"
        以下是该片段中连续的 {N} 帧画面。
        对每帧用一到两句话描述其具体画面内容。
        重点关注: 动作、物体变化、文字信息、人物表情。
        不要重复片段整体描述，聚焦每帧的区分性信息。
        '''
        """

    def _build_l1_video(self, l2_children: List[L2Node]) -> L1Node:
        """Step 4: 同文本模式，聚合 L2 描述 → L1 摘要"""
```

**依赖**: tree_index, embeddings, llm_client, opencv-python（帧提取）

---

### 1.6 recursive_retriever.py — TRM 递归检索器

**文件**: `video_tree_trm/recursive_retriever.py`
**职责**: 核心可训练模型。Cross-Attention 节点选择 + MLP 推理 + ACT halt。

#### 1.6.1 CrossAttentionSelector

```python
class CrossAttentionSelector(nn.Module):
    """跨层节点选择器（共享，用于 L1/L2/L3 三个阶段）"""

    def __init__(self, embed_dim: int, num_heads: int):
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)
        self.W_o = Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(
        self, state: Tensor, candidates: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            state:      [B, D] — 当前 q+z 融合状态
            candidates: [B, N, D] — 该层候选节点嵌入
        Returns:
            selected_info: [B, D] — attention 加权节点信息（可微）
            attn_weights:  [B, N] — 归一化注意力权重（用于 nav loss）
            selected_idx:  [B] — argmax 节点索引（用于路径记录）
        """
        B, N, D = candidates.shape

        Q = self.W_q(state).unsqueeze(1)            # [B, 1, D]
        K = self.W_k(candidates)                     # [B, N, D]
        V = self.W_v(candidates)                     # [B, N, D]

        # reshape → multi-head
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        # scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # [B, H, 1, d]
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, D)
        selected_info = self.W_o(attn_out).squeeze(1)        # [B, D]

        # 注意力权重（对 head 维度平均，用于 loss 和可解释性）
        raw_scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = raw_scores.mean(dim=1).squeeze(1).softmax(dim=-1)  # [B, N]
        selected_idx = attn_weights.argmax(dim=-1)                         # [B]

        return selected_info, attn_weights, selected_idx
```

#### 1.6.2 ReasoningModule（L-level）

```python
class ReasoningBlock(nn.Module):
    """单层 MLP 推理块"""
    def __init__(self, dim: int, expansion: float):
        self.norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * expansion))

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.ffn(x))  # [B, D] → [B, D]


class ReasoningModule(nn.Module):
    """L-level 推理模块（多层 MLP，共享权重跨层级）"""
    def __init__(self, dim: int, L_layers: int, expansion: float):
        self.blocks = ModuleList([ReasoningBlock(dim, expansion) for _ in range(L_layers)])

    def forward(self, z: Tensor, injection: Tensor) -> Tensor:
        """
        Args:
            z:         [B, D] — 当前潜在状态
            injection: [B, D] — 注入信息 (selected_info + q)
        Returns:
            z_new: [B, D]
        """
        h = z + injection
        for block in self.blocks:
            h = block(h)
        return h
```

#### 1.6.3 RecursiveRetriever

```python
class RecursiveRetriever(nn.Module):
    """TRM 递归检索器主模型"""

    def __init__(self, config: RetrieverConfig):
        self.selector = CrossAttentionSelector(config.embed_dim, config.num_heads)
        self.L_level = ReasoningModule(config.embed_dim, config.L_layers, config.ffn_expansion)
        self.q_head = Linear(config.embed_dim, 1)  # ACT halt head
        self.L_cycles = config.L_cycles
        self.max_rounds = config.max_rounds

        # q_head 初始化为倾向"继续"（bias = -5 → sigmoid ≈ 0）
        with torch.no_grad():
            self.q_head.bias.fill_(-5.0)

    def forward(
        self, q: Tensor, tree: TreeIndex, return_internals: bool = False
    ) -> Dict[str, Any]:
        """
        训练/推理统一入口。
        Args:
            q:    [B, D] — 查询嵌入（来自冻结 text_embed）
            tree: TreeIndex — 预构建树索引
            return_internals: 是否返回中间状态（用于 loss 计算）
        Returns:
            {
                "paths": List[Tuple[int, int, int]],
                "num_rounds": int,
                "z_final": Tensor [B, D],
                # return_internals=True 时额外返回:
                "attn_weights_per_step": List[Tensor],  # 每步 [B, N]
                "halt_logits": List[Tensor],             # 每轮 [B, 1]
            }
        """
        z = q.clone()                   # [B, D]
        paths = []
        attn_weights_all = []
        halt_logits_all = []

        for round_idx in range(self.max_rounds):
            path, z, step_attns = self._traverse_one_path(q, z, tree)
            paths.append(path)
            attn_weights_all.extend(step_attns)

            halt_logit = self.q_head(z)  # [B, 1]
            halt_logits_all.append(halt_logit)

            if not self.training and halt_logit.item() > 0 and round_idx > 0:
                break

        result = {
            "paths": paths,
            "num_rounds": len(paths),
            "z_final": z,
        }
        if return_internals:
            result["attn_weights_per_step"] = attn_weights_all
            result["halt_logits"] = halt_logits_all
        return result

    def _traverse_one_path(
        self, q: Tensor, z: Tensor, tree: TreeIndex
    ) -> Tuple[Tuple[int, int, int], Tensor, List[Tensor]]:
        """单次 L1 → L2 → L3 遍历"""
        step_attns = []

        # Phase 1: L1
        M_L1 = torch.tensor(tree.l1_embeddings(), device=q.device)  # [N1, D]
        k1, z, attn_w = self._select_and_reason(q, z, M_L1.unsqueeze(0))
        step_attns.append(attn_w)

        # Phase 2: L2 (k1 的子节点)
        M_L2 = torch.tensor(tree.l2_embeddings_of(k1), device=q.device)  # [N2, D]
        k2, z, attn_w = self._select_and_reason(q, z, M_L2.unsqueeze(0))
        step_attns.append(attn_w)

        # Phase 3: L3 (k2 的子节点)
        M_L3 = torch.tensor(tree.l3_embeddings_of(k1, k2), device=q.device)  # [N3, D]
        k3, z, attn_w = self._select_and_reason(q, z, M_L3.unsqueeze(0))
        step_attns.append(attn_w)

        return (k1, k2, k3), z, step_attns

    def _select_and_reason(
        self, q: Tensor, z: Tensor, M: Tensor
    ) -> Tuple[int, Tensor, Tensor]:
        """
        单层: Cross-Attention 选择 + L_cycles 内循环推理
        Args:
            q: [B, D], z: [B, D], M: [B, N, D]
        Returns:
            k_star: int, z_new: [B, D], attn_weights: [B, N]
        """
        state = q + z
        selected_info, attn_weights, selected_idx = self.selector(state, M)

        z = z + selected_info

        for _ in range(self.L_cycles):
            z = self.L_level(z, selected_info + q)

        return selected_idx.item(), z, attn_weights
```

**训练 vs 推理行为差异**:

| 行为 | 训练 | 推理 |
|------|------|------|
| 多轮循环 | 固定跑 max_rounds 轮 | halt_logit > 0 提前停止 |
| 梯度 | 全部可微 | no_grad |
| 返回值 | 含 attn_weights + halt_logits | 仅 paths + z_final |

**依赖**: torch, tree_index

---

### 1.7 losses.py — 损失函数

**文件**: `video_tree_trm/losses.py`
**职责**: 导航损失（cross-entropy）+ ACT halt 损失（Q-learning）。

```python
class NavigationLoss(nn.Module):
    """导航监督损失：推动 attn_weights 指向正确节点"""

    def forward(
        self, attn_weights_list: List[Tensor], gt_path: Tuple[int, int, int]
    ) -> Tensor:
        """
        Args:
            attn_weights_list: [attn_l1, attn_l2, attn_l3]，每个 [B, N]
            gt_path: (gt_l1_idx, gt_l2_idx, gt_l3_idx)
        Returns:
            loss: scalar
        """
        loss = 0
        for attn_w, gt_idx in zip(attn_weights_list, gt_path):
            target = torch.tensor([gt_idx], device=attn_w.device)
            log_probs = attn_w.log()                 # [B, N]
            loss += F.nll_loss(log_probs, target)     # cross-entropy
        return loss / 3  # 三层平均


class ACTLoss(nn.Module):
    """ACT halt Q-learning 损失"""

    def __init__(self, lambda_step: float = 0.1, gamma: float = 0.9):
        self.lambda_step = lambda_step
        self.gamma = gamma

    def forward(
        self,
        halt_logits: List[Tensor],     # 每轮 [B, 1]
        answer_qualities: List[float],  # 每轮累积的答案质量 (0~1)
    ) -> Tensor:
        """
        Q-learning target:
          若在第 t 轮停止 → Q_halt = quality_t
          若继续          → Q_continue = γ * max(Q_{t+1}) - λ
        """
        loss = 0
        n = len(halt_logits)
        for t in range(n):
            halt_q = answer_qualities[t]
            if t < n - 1:
                continue_q = self.gamma * answer_qualities[t + 1] - self.lambda_step
            else:
                continue_q = halt_q - self.lambda_step  # 最后一轮，继续无意义

            # 目标: halt_logit > 0 当 halt_q > continue_q
            target = 1.0 if halt_q >= continue_q else 0.0
            pred = torch.sigmoid(halt_logits[t])
            loss += F.binary_cross_entropy(pred, torch.tensor([[target]], device=pred.device))

        return loss / n
```

**依赖**: torch

---

### 1.8 answer_generator.py — 答案生成

**文件**: `video_tree_trm/answer_generator.py`
**职责**: 根据检索结果组装 context，调用 LLM/VLM 生成最终答案。

```python
@dataclass
class RetrievalResult:
    """检索器输出的结构化结果"""
    query: str
    paths: List[Tuple[int, int, int]]
    num_rounds: int

class AnswerGenerator:
    def __init__(self, llm: LLMClient, vlm: LLMClient):
        self.llm = llm
        self.vlm = vlm

    def generate(self, query: str, result: RetrievalResult, tree: TreeIndex) -> str:
        """
        根据模态分发:
          文本 → LLM(query, raw_text_chunks)
          视频 → VLM(query, frame_images + captions)
        """
        nodes = [tree.get_node(*path) for path in result.paths]

        if tree.metadata.modality == "text":
            context = "\n---\n".join(n.raw_content for n in nodes if n.raw_content)
            return self.llm.chat(
                f"根据以下上下文回答问题。\n\n上下文:\n{context}\n\n问题: {query}"
            )
        else:
            frames = [n.frame_path for n in nodes if n.frame_path]
            captions = [n.description for n in nodes]
            caption_text = "\n".join(f"- {c}" for c in captions)
            return self.vlm.chat_with_images(
                f"根据以下关键帧回答问题。\n帧描述:\n{caption_text}\n\n问题: {query}",
                images=frames,
            )
```

**依赖**: tree_index, llm_client

---

### 1.9 pipeline.py — 端到端管线

**文件**: `video_tree_trm/pipeline.py`
**职责**: 串联 预处理 → 检索 → 生成 的完整推理流程。

```python
class Pipeline:
    """端到端推理管线"""

    def __init__(self, config: Config):
        self.embed_model = EmbeddingModel(config.embed.model_name, config.embed.device)
        self.llm = LLMClient(config.llm.backend, config.llm.api_key, config.llm.model)
        self.vlm = LLMClient(config.vlm.backend, config.vlm.api_key, config.vlm.model)
        self.retriever = RecursiveRetriever(config.retriever)
        self.retriever.load_state_dict(torch.load(config.retriever.checkpoint))
        self.retriever.eval()
        self.generator = AnswerGenerator(self.llm, self.vlm)

    def build_index(self, source_path: str, modality: str) -> TreeIndex:
        """构建并缓存 TreeIndex"""
        if modality == "text":
            builder = TextTreeBuilder(self.embed_model, self.llm, self.config.tree)
            with open(source_path) as f:
                return builder.build(f.read(), source_path)
        else:
            builder = VideoTreeBuilder(self.embed_model, self.vlm, self.config.tree)
            return builder.build(source_path)

    def query(self, question: str, tree: TreeIndex) -> str:
        """问答: question → answer"""
        q = torch.tensor(self.embed_model.embed(question), device=self.config.embed.device)
        with torch.no_grad():
            result = self.retriever(q.unsqueeze(0), tree)
        retrieval_result = RetrievalResult(
            query=question, paths=result["paths"], num_rounds=result["num_rounds"]
        )
        return self.generator.generate(question, retrieval_result, tree)
```

**依赖**: 所有其他模块

---

### 1.10 config.py — 配置管理

**文件**: `video_tree_trm/config.py`
**职责**: 所有超参数的 dataclass 类型定义（无默认值）+ 多源加载。

#### 设计原则

- **Dataclass 无默认值**: 纯类型定义 + 结构化访问，YAML 必须写全，漏写即报错。
- **三层优先级**: `CLI args > .env > YAML`，高优先级覆盖低优先级。
- **统一归口**: 无论来源，最终构造唯一 `Config` dataclass 对象，代码只与 dataclass 交互。
- **敏感信息隔离**: `api_key` 等敏感字段只写在 `.env` 中，不进 YAML 和代码。

#### 加载流程

```
Step 1: 读取 YAML → base dict（全量非敏感配置）
Step 2: 读取 .env  → 覆盖 dict 中对应字段（api_key 等敏感信息）
Step 3: 解析 CLI args → 最终覆盖 dict 中对应字段
Step 4: dict → Config dataclass（校验完整性，缺字段直接报错）
```

#### Dataclass 定义

```python
@dataclass
class TreeConfig:
    # 文本模式
    max_paragraphs_per_l2: int              # 每个 L2 节点包含的最大段落数
    # 视频模式
    l1_segment_duration: float              # L1 段时长（秒）
    l2_clip_duration: float                 # L2 clip 时长（秒）
    l3_fps: float                           # L3 帧提取频率
    l2_representative_frames: int           # L2 VLM 描述用的代表帧数
    # 通用
    cache_dir: str                          # TreeIndex 缓存目录

@dataclass
class EmbedConfig:
    model_name: str                         # 嵌入模型名称
    embed_dim: int                          # 嵌入维度 D
    device: str                             # "cuda" | "cpu"

@dataclass
class LLMConfig:
    backend: str                            # "qwen" | "openai" | "ollama"
    api_key: str                            # 从 .env 加载，不写入 YAML
    model: str                              # 模型名称
    api_url: str                            # API 端点 URL
    max_tokens: int                         # 最大生成 token 数
    temperature: float                      # 采样温度

@dataclass
class VLMConfig:
    backend: str                            # "qwen" | "openai" | "ollama"
    api_key: str                            # 从 .env 加载，不写入 YAML
    model: str                              # 模型名称
    api_url: str                            # API 端点 URL
    max_tokens: int                         # 最大生成 token 数
    temperature: float                      # 采样温度

@dataclass
class RetrieverConfig:
    embed_dim: int                          # 嵌入维度（须与 EmbedConfig.embed_dim 一致）
    num_heads: int                          # Cross-Attention 头数
    L_layers: int                           # ReasoningModule 层数
    L_cycles: int                           # 每级推理迭代次数
    max_rounds: int                         # ACT 最大遍历轮次
    ffn_expansion: float                    # SwiGLU 扩展比
    checkpoint: Optional[str]               # 训练好的模型权重路径（推理时必填）

@dataclass
class TrainConfig:
    lr: float                               # 学习率
    weight_decay: float                     # 权重衰减
    batch_size: int                         # 批大小
    max_epochs_phase1: int                  # Phase 1 导航训练轮数
    max_epochs_phase2: int                  # Phase 2 ACT 训练轮数
    nav_loss_weight: float                  # 导航损失权重
    act_loss_weight: float                  # ACT 损失权重
    act_lambda_step: float                  # ACT 步数惩罚系数
    act_gamma: float                        # ACT 折扣因子
    eval_interval: int                      # 每 N epoch 评估一次
    save_dir: str                           # 模型权重保存目录
    dataset: str                            # "longbench" | "narrativeqa" | "videomme"
    dataset_path: str                       # 数据集路径

@dataclass
class Config:
    tree: TreeConfig
    embed: EmbedConfig
    llm: LLMConfig
    vlm: VLMConfig
    retriever: RetrieverConfig
    train: TrainConfig

    @classmethod
    def load(cls, yaml_path: str, cli_args: Optional[dict] = None) -> "Config":
        """
        三层合并加载:
          1. 读取 YAML → base dict
          2. 读取 .env  → 覆盖 api_key 等敏感字段
          3. cli_args   → 最终覆盖
          4. dict → Config（缺字段报 TypeError）
        """
        ...
```

#### 文件分工

| 文件 | 内容 | 提交到 Git |
|------|------|-----------|
| `config/default.yaml` | 全量非敏感配置（必须写全所有字段） | 是 |
| `.env` | 敏感信息（api_key 等） | 否 |
| `.env.example` | `.env` 模板（值留空） | 是 |

#### YAML 示例 (`config/default.yaml`)

```yaml
tree:
  max_paragraphs_per_l2: 5
  l1_segment_duration: 600.0
  l2_clip_duration: 20.0
  l3_fps: 1.0
  l2_representative_frames: 3
  cache_dir: "cache/trees"

embed:
  model_name: "BAAI/bge-base-zh-v1.5"
  embed_dim: 768
  device: "cuda"

llm:
  backend: "qwen"
  model: "qwen-plus"
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  max_tokens: 256
  temperature: 0.1
  # api_key: 从 .env 加载，此处不写

vlm:
  backend: "qwen"
  model: "qwen-vl-plus"
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  max_tokens: 256
  temperature: 0.1
  # api_key: 从 .env 加载，此处不写

retriever:
  embed_dim: 768
  num_heads: 4
  L_layers: 2
  L_cycles: 4
  max_rounds: 5
  ffn_expansion: 2.0
  checkpoint: null

train:
  lr: 1.0e-4
  weight_decay: 1.0e-5
  batch_size: 1
  max_epochs_phase1: 30
  max_epochs_phase2: 20
  nav_loss_weight: 1.0
  act_loss_weight: 0.1
  act_lambda_step: 0.1
  act_gamma: 0.9
  eval_interval: 5
  save_dir: "checkpoints"
  dataset: "longbench"
  dataset_path: "data/longbench"
```

#### .env 示例

```bash
# .env — 敏感信息，不提交到 Git
LLM_API_KEY=sk-xxx
VLM_API_KEY=sk-xxx
```

**依赖**: dataclasses, yaml, python-dotenv

---

## 2. 训练管线

**文件**: `train.py`

### 2.1 数据准备

```python
def prepare_training_data(config: Config) -> List[Dict]:
    """
    离线预处理:
      1. 加载 QA 数据集（LongBench / NarrativeQA）
      2. 为每个文档构建 TreeIndex（缓存到 cache_dir）
      3. 推导每个 QA 对的 ground truth 路径
    Returns:
        [{"query": str, "tree": TreeIndex, "gt_path": (l1, l2, l3), "answer": str}, ...]
    """
```

### 2.2 Ground Truth 路径推导

```python
def find_gt_path_text(tree: TreeIndex, answer: str) -> Optional[Tuple[int, int, int]]:
    """
    文本模式: 找到与答案文本重叠度最高的 L3 节点
    评分: F1(L3.raw_content, answer) — token 级别
    返回: (l1_idx, l2_idx, l3_idx) 或 None
    """
    best_score, best_path = 0, None
    for i, l1 in enumerate(tree.roots):
        for j, l2 in enumerate(l1.children):
            for k, l3 in enumerate(l2.children):
                score = token_f1(l3.raw_content, answer)
                if score > best_score:
                    best_score = score
                    best_path = (i, j, k)
    return best_path


def find_gt_path_video(tree: TreeIndex, timestamp: float) -> Optional[Tuple[int, int, int]]:
    """
    视频模式: 找到最接近目标时间戳的 L3 帧
    """
    for i, l1 in enumerate(tree.roots):
        if l1.time_range[0] <= timestamp <= l1.time_range[1]:
            for j, l2 in enumerate(l1.children):
                if l2.time_range[0] <= timestamp <= l2.time_range[1]:
                    k = min(range(len(l2.children)),
                            key=lambda k: abs(l2.children[k].timestamp - timestamp))
                    return (i, j, k)
    return None
```

### 2.3 两阶段训练策略

```
Phase 1: 导航训练（单轮, max_rounds=1）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  目标: 训练 Selector + L_level 正确导航到目标节点
  损失: NavigationLoss (cross-entropy on attn_weights)
  可训练: CrossAttentionSelector, ReasoningModule
  冻结:   text_embed, q_head, TreeIndex embeddings

Phase 2: ACT 训练（多轮, max_rounds=5）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  目标: 训练 q_head 判断何时停止检索
  损失: NavigationLoss + λ * ACTLoss
  可训练: 全部（Selector + L_level + q_head）
  冻结:   text_embed, TreeIndex embeddings
  ACT reward: answer_quality (F1/EM) - λ_step * rounds
```

### 2.4 训练循环伪代码

```python
def train(config: Config):
    # ── 初始化 ──
    embed_model = EmbeddingModel(config.embed.model_name, config.embed.device)
    retriever = RecursiveRetriever(config.retriever).to(config.embed.device)
    nav_loss_fn = NavigationLoss()
    act_loss_fn = ACTLoss(config.train.act_lambda_step, config.train.act_gamma)

    dataset = prepare_training_data(config)
    optimizer = AdamW(retriever.parameters(), lr=config.train.lr)

    # ── Phase 1: 导航训练 ──
    retriever.max_rounds = 1
    for epoch in range(config.train.max_epochs_phase1):
        for sample in dataset:
            q = embed_model.embed_tensor(sample["query"]).to(device)  # [1, D]
            tree = sample["tree"]
            gt_path = sample["gt_path"]

            result = retriever(q, tree, return_internals=True)
            # result["attn_weights_per_step"] = [attn_l1, attn_l2, attn_l3]
            loss = nav_loss_fn(result["attn_weights_per_step"][:3], gt_path)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ── Phase 2: ACT 训练 ──
    retriever.max_rounds = config.retriever.max_rounds
    llm = LLMClient(config.llm.backend, config.llm.api_key, config.llm.model)
    generator = AnswerGenerator(llm, None)

    for epoch in range(config.train.max_epochs_phase2):
        for sample in dataset:
            q = embed_model.embed_tensor(sample["query"]).to(device)
            result = retriever(q, sample["tree"], return_internals=True)

            # 每轮计算答案质量
            qualities = []
            for round_idx in range(result["num_rounds"]):
                paths_so_far = result["paths"][:round_idx + 1]
                nodes = [sample["tree"].get_node(*p) for p in paths_so_far]
                context = "\n".join(n.raw_content for n in nodes if n.raw_content)
                answer = llm.chat(f"上下文: {context}\n问题: {sample['query']}")
                quality = token_f1(answer, sample["answer"])
                qualities.append(quality)

            # 导航 loss（仅第一轮）
            loss_nav = nav_loss_fn(result["attn_weights_per_step"][:3], sample["gt_path"])
            # ACT loss
            loss_act = act_loss_fn(result["halt_logits"], qualities)
            # 总损失
            loss = loss_nav + config.train.act_loss_weight * loss_act

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 3. 实验计划

### 3.1 数据集

| 数据集 | 模态 | 样本量 | 任务类型 | 优先级 |
|--------|------|--------|----------|--------|
| LongBench | 文本 | ~5K | 长文本 QA | P0 (首发) |
| NarrativeQA | 文本 | ~30K | 叙事理解 QA | P1 |
| VideoMME | 视频 | ~2K | 视频 QA (多选) | P2 |

### 3.2 评估指标

| 指标 | 适用 | 计算方式 |
|------|------|----------|
| EM (Exact Match) | 文本 QA | 标准化后精确匹配 |
| F1 | 文本 QA | token 级 precision/recall |
| Accuracy | 视频 QA | 选项匹配正确率 |
| Avg Rounds | 全部 | 平均检索轮次（衡量效率） |
| Nav Accuracy | 全部 | 第一轮 L1/L2/L3 各层命中率 |

### 3.3 Baselines

| 方法 | 描述 |
|------|------|
| BM25 + LLM | 传统稀疏检索 baseline |
| Dense Retrieval + LLM | BGE 向量检索 + rerank |
| PageIndex (原论文) | 无 TRM 的树状导航 (cosine routing, 无推理模块) |
| Tree-TRM (原论文) | 原始 tree_trm.py 实现 |

### 3.4 消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| A1 | Cross-Attention vs Cosine 路由 | 验证 CA 选择器的增益 |
| A2 | L_cycles = {1, 2, 4, 8} | 推理深度对准确率的影响 |
| A3 | L_layers = {1, 2, 4} | 推理模块复杂度 |
| A4 | max_rounds = {1, 3, 5} | 多轮检索的边际收益 |
| A5 | 有/无 ACT halt | ACT 机制对效率的贡献 |
| A6 | num_heads = {1, 4, 8} | 注意力头数的影响 |

---

## 4. 文件结构与依赖

### 4.1 目录树

```
Video-Tree-TRM/
├── video_tree_trm/                   # 主包
│   ├── __init__.py
│   ├── config.py                     # §1.10 配置管理
│   ├── tree_index.py                 # §1.1  统一数据结构
│   ├── embeddings.py                 # §1.2  嵌入服务
│   ├── llm_client.py                 # §1.3  LLM/VLM 客户端
│   ├── text_tree_builder.py          # §1.4  文本树构建
│   ├── video_tree_builder.py         # §1.5  视频树构建
│   ├── recursive_retriever.py        # §1.6  TRM 递归检索器
│   ├── losses.py                     # §1.7  损失函数
│   ├── answer_generator.py           # §1.8  答案生成
│   └── pipeline.py                   # §1.9  端到端管线
├── utils/
│   ├── __init__.py
│   └── logger_system.py              # 日志系统 (log_msg, ensure, log_exception)
├── config/
│   └── default.yaml                  # 默认配置
├── tests/
│   ├── unit/
│   │   ├── test_tree_index.py
│   │   ├── test_recursive_retriever.py
│   │   └── test_losses.py
│   ├── integration/
│   │   ├── test_text_tree_builder.py
│   │   └── test_pipeline.py
│   └── outputs/                      # Agent 测试 MD 输出
├── data/                             # 数据集（不提交）
├── cache/                            # TreeIndex 缓存（不提交）
├── checkpoints/                      # 模型权重（不提交）
├── logs/                             # 运行日志（不提交）
├── train.py                          # §2 训练入口
├── main.py                           # 推理/演示入口
├── docs/
│   ├── architecture.md               # 架构设计（理念层）
│   └── TD.md                         # 本文档（实现层）
├── .env                              # API 密钥（不提交）
├── .env.example                      # 环境变量模板
└── requirements.txt
```

### 4.2 模块依赖关系

```
config.py ← (所有模块都依赖)

embeddings.py ← text_tree_builder.py
              ← video_tree_builder.py
              ← pipeline.py

llm_client.py ← text_tree_builder.py
              ← video_tree_builder.py
              ← answer_generator.py
              ← pipeline.py

tree_index.py ← text_tree_builder.py
              ← video_tree_builder.py
              ← recursive_retriever.py
              ← answer_generator.py
              ← pipeline.py

recursive_retriever.py ← pipeline.py
                       ← train.py

losses.py ← train.py

answer_generator.py ← pipeline.py
                    ← train.py (Phase 2, 计算 answer quality)
```

### 4.3 Python 依赖

```
# 核心
torch>=2.0
sentence-transformers>=2.2
numpy

# LLM/VLM
openai>=1.0              # 兼容 Qwen/OpenAI/Ollama 接口
python-dotenv

# 视频处理
opencv-python

# 配置
pyyaml

# 测试
pytest
pytest-cov

# 代码质量
ruff
```

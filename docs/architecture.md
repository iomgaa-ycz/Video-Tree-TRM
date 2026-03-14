# Video-Tree-TRM 统一架构设计

## 1. 核心定位

结合 TRM 递归推理 + PageIndex 树状导航的 **模态无关 RAG 系统**。

```
设计原则:
  - 检索阶段模态无关（全文本嵌入空间）
  - 模态差异封装在两端（预处理 + 答案生成）
  - TRM ACT 机制控制动态检索深度
  - 树深度固定 3 层，检索轮次动态
```

---

## 2. 系统总览

```
 ┌──────────────────── 预处理（离线，一次性） ────────────────────┐
 │                                                               │
 │  原始输入                   统一表示                           │
 │  ┌──────────┐              ┌──────────────────────┐           │
 │  │ 长文本    │──TextTree──→│                      │           │
 │  └──────────┘   Builder    │     TreeIndex         │           │
 │  ┌──────────┐              │  (全文本嵌入, [D])    │           │
 │  │ 长视频    │──VideoTree─→│                      │           │
 │  └──────────┘   Builder    └──────────┬───────────┘           │
 │                                       │                       │
 └───────────────────────────────────────│───────────────────────┘
                                         │
 ┌──────────────────── 检索（在线） ─────│───────────────────────┐
 │                                       ↓                       │
 │  query ──text_embed──→ q ──→ RecursiveRetriever               │
 │                              │  TRM 双循环推理                │
 │                              │  ACT halt 动态停止             │
 │                              │  多轮 root-to-leaf 遍历        │
 │                              └──→ List[NodePath]              │
 │                                     │                         │
 └─────────────────────────────────────│─────────────────────────┘
                                       │
 ┌──────────────────── 生成（在线） ────│─────────────────────────┐
 │                                      ↓                        │
 │  文本模式: LLM(query, raw_text_chunks)  → answer              │
 │  视频模式: VLM(query, frame_images)     → answer              │
 │                                                               │
 └───────────────────────────────────────────────────────────────┘
```

---

## 3. TreeIndex：统一数据结构

三层固定深度，文本/视频同构。

```
TreeIndex
├── metadata: IndexMeta          # 来源、嵌入器版本、创建时间
├── embed_dim: int               # 嵌入维度 D
└── roots: List[L1Node]          # L1 节点列表（树的根层）

L1Node (段/章)
├── id: str
├── summary: str                 # 聚合自子 L2 描述 (2-3句)
├── embedding: ndarray [D]       # text_embed(summary)
├── time_range: (float, float)   # 仅视频模式
└── children: List[L2Node]

L2Node (片段/小节)
├── id: str
├── description: str             # 直接从原始内容生成 (1-2句)
├── embedding: ndarray [D]       # text_embed(description)
├── time_range: (float, float)   # 仅视频模式
└── children: List[L3Node]

L3Node (帧/文本块)
├── id: str
├── description: str             # 视频=VLM帧描述(继承L2上下文), 文本=原始段落文本
├── embedding: ndarray [D]       # text_embed(description)
├── raw_content: str             # 原始文本块（文本模式）
├── frame_path: Optional[str]    # 帧图像路径（视频模式）
└── timestamp: Optional[float]   # 帧时间戳（视频模式）
```

**关键设计**：`embedding` 字段全部来自同一个 `text_embed()` 函数，不存在跨模态嵌入空间问题。`raw_content` / `frame_path` 仅供答案生成阶段使用，检索器不关心。

---

## 4. 预处理管线

### 4.1 摘要粒度规范

各层摘要的唯一职责是**生成好的路由嵌入**——让 `(q+z)·M^T/√D` 能选对节点。

| 层级 | 路由功能 | 摘要目标 | 粒度 |
|------|----------|----------|------|
| L1 | 粗路由：选主题区域 | "这个区域**关于什么**" | 2-3句：主题 + 覆盖范围 + 关键实体 |
| L2 | 细聚焦：选具体片段 | "这个片段**发生了什么**" | 1-2句：具体事件/内容 + 区分性细节 |
| L3(文本) | 精确定位：选文本块 | 段落原文 | 原始段落文本，不做摘要 |
| L3(视频) | 精确定位：选帧 | "这帧画面的**具体内容**" | 1-2句：继承L2上下文，聚焦区分性视觉信息 |

```
关键原则:
  - L1 要宽: 涵盖其下所有 L2 的语义范围，让相关 query 能"进对门"
  - L2 要窄: 只描述自己，与同级兄弟节点形成区分
  - L3 要具体: 提供精确定位的细节信息
```

**文本示例**：
```
L1: "第三章讨论了用户认证系统的设计，涵盖OAuth流程、JWT令牌管理和权限控制。"
  L2: "OAuth 2.0授权码流程的实现，包括重定向和回调处理。"
  L2: "JWT令牌的生成、验证和刷新机制。"
  L2: "基于角色的权限控制模型（RBAC）。"
```

**视频示例**：
```
L1: "厨师在厨房制作意大利面，从准备食材到最终装盘。"
  L2: "切洋葱、蒜末和番茄，准备酱料食材。"
    L3: "[准备酱料食材中] 厨师左手按住洋葱，右手快速切丁，砧板上已有切好的蒜末。"
    L3: "[准备酱料食材中] 厨师将番茄放入沸水中烫皮，锅中冒出蒸汽。"
  L2: "煮面条并制作番茄肉酱。"
  L2: "装盘摆盘并撒上帕尔马干酪。"
```

### 4.2 构建顺序：L2 轴心策略

两种模态共享同一构建原则：**L2 为轴心，向下展开 L3，向上聚合 L1**。

```
构建依赖关系:

  L3 需要 L2 上下文才能生成高质量描述（视频尤为关键）
  L1 摘要应从 L2 聚合，确保覆盖面完整
  → 循环依赖 → 解法: L2 不依赖 L3，而是从原始内容独立生成

构建顺序:

  Step 1: 结构切分 → 确定 L1/L2 边界
  Step 2: L2 先行 → 从原始内容直接生成 L2 描述
  Step 3: L3 向下 → 注入 L2 上下文，生成细粒度描述
  Step 4: L1 向上 → 聚合 L2 描述，生成粗粒度摘要
```

```
         Step 4: 聚合
            ↑
  ┌─────── L1 ───────┐      "第三章讨论了认证系统..."
  │                   │       ← LLM("总结这些片段: " + L2_descriptions)
  │   Step 2: 先行    │
  ├── L2 ── L2 ── L2 ┤      "OAuth流程的实现..."
  │    │    │    │     │      ← 文本: LLM 摘要段落组
  │    ↓    ↓    ↓     │        视频: VLM 看少量代表帧
  │   L3s  L3s  L3s    │
  │   Step 3: 向下     │      "厨师左手按住洋葱..."
  └────────────────────┘      ← 注入 L2 上下文后生成
```

### 4.3 TextTreeBuilder

```
输入: 长文本文档
输出: TreeIndex（全部 embedding=None）

Pipeline:
  Step 1 — 结构切分
     有 ToC → 解析章节层级 → L1/L2 边界
     无 ToC → LLM 语义分段 (一次性调用)

  Step 2 — L2 先行
     L2: 段落组 → LLM 生成摘要 (1-2句)
         摘要目标: 与兄弟 L2 形成区分
         （batch_chat 并发生成所有 L2 摘要）

  Step 3 — L3 向下
     L3: 原始段落文本直接复用
         存储 raw_content = description = 原始文本
         （文本模式 L3 不需要 L2 上下文，不调用 LLM）

  Step 4 — L1 向上聚合
     L1: LLM("总结这些小节: " + 所有子 L2 摘要) → 摘要
         摘要目标: 覆盖下属所有 L2 的语义范围 (2-3句)

  Step 5 — 序列化 → TreeIndex（JSON，无 embedding）

  ⚠ 延迟 Embedding: 所有节点 embedding=None
     首次检索时由 Pipeline._embed_tree() → tree.embed_all() 统一填充
```

### 4.4 VideoTreeBuilder

**状态**: ✅ 已实现（`video_tree_trm/video_tree_builder.py`）

```
输入: 长视频文件路径 或 YouTube URL
输出: TreeIndex（全部 embedding=None）

Pipeline（ThreadPoolExecutor 异步事件循环）:
  Step 0 — 输入类型判断
     本地文件: 直接使用 OpenCV 读取
     YouTube URL: yt-dlp -g 获取 CDN 直链 + yt-dlp --dump-json 获取时长

  Step 1 — 时间切分（固定步长）
     本地: cv2 读取总时长
     HTTP 流: 使用 yt-dlp 元数据时长（duration_hint，避免 cv2 流上不可靠）
     固定步长 l1_segment_duration=600s → L1 区间列表
     每个 L1 区间 → 等分 l2_clip_duration=60s → L2 clips

  Step 2 — 预计算任务图
     收集所有 L2 任务 + 记录每个 L1 的 L2 数量（l2_counts）

  Step 3 — 一次性提交所有 L2 任务（非阻塞）
     ThreadPoolExecutor(max_workers=concurrency)
     每个 L2 clip: 均匀 seek l2_representative_frames=10 帧 → VLM → 1-2句描述

  Step 4 — 事件循环（cfwait FIRST_COMPLETED）
     L2 完成 → 立即提交 L3 任务（_build_l3_task，线程安全）
       L3: 按 l3_fps 提取全量帧（持久化缓存），注入 L2 上下文，VLM 批量帧描述（JSON）
           降级路径: JSON 解析失败 → 逐帧 VLM 调用
     L3 完成 → 检查该 L1 的所有 L2 是否就绪 → 触发 L1 任务
       L1: 拼接所有 L2 描述 → vlm.chat() → 2-3句摘要
     主线程单线程操作 l1_l2_buckets，无竞争，无需 Lock

  Step 5 — 有序重建 l1_nodes，组装 TreeIndex（写 IndexMeta + 日志）

  ⚠ 延迟 Embedding: 所有节点 embedding=None
     首次检索时由 Pipeline._embed_tree() → tree.embed_all() 统一填充
```

**L2 代表帧采样**（均匀 seek，首尾均包含）：

```python
step = (end_sec - start_sec) / (n_rep - 1)
timestamps = [start_sec + i * step for i in range(n_rep)]
# → 直接 cap.set(CAP_PROP_POS_MSEC, ts * 1000) 提取，缓存为 l2_{ts:.3f}.jpg
```

**L3 帧描述 prompt**（继承 L2 上下文 + 要求 JSON 输出）：

```python
prompt = (
    '该片段的整体内容: "{l2_description}"\n'
    "以下是该片段中连续的 {n} 帧画面。\n"
    "对每帧用一到两句话描述其具体画面内容。\n"
    "重点关注: 动作、物体变化、文字信息、人物表情。\n"
    "不要重复片段整体描述，聚焦每帧的区分性信息。\n"
    '只返回 JSON 数组，格式: ["帧1描述", "帧2描述", ...]，不要其他内容。'
)
descs = VLM(frames_batch, prompt)  # 主路径：一次调用，JSON 解析
# 降级：json.loads 失败 → 逐帧调用
```

**L1 摘要 prompt**：

```python
l2_texts = "\n".join(f"- {node.description}" for node in l2_children)
prompt = f"以下是一个视频段落中各片段的描述:\n{l2_texts}\n用2-3句话总结该段落的整体内容，涵盖所有片段的主题。"
l1_summary = vlm.chat(prompt)  # 纯文本调用
```

---

## 5. RecursiveRetriever：TRM 式递归检索

### 5.1 核心算法

节点选择使用 **Cross-Attention**（学习 W_q/W_k/W_v/W_o 投影），替代简单 cosine 路由。
L_level 推理模块使用 **MLP-based**（RMSNorm + SwiGLU），操作对象为向量 `[B, D]`。
三个可学习组件（CrossAttentionSelector, ReasoningModule, q_head）**跨层级共享权重**。

```
┌─────────────────────────────────────────────────────────────┐
│  RecursiveRetriever                                          │
│                                                              │
│  可训练组件（共享权重）:                                       │
│    selector = CrossAttentionSelector(W_q, W_k, W_v, W_o)    │
│    L_level  = ReasoningModule(RMSNorm + SwiGLU, L_layers层)  │
│    q_head   = Linear(D, 1)                                   │
│                                                              │
│  输入: query (str), tree (TreeIndex)                         │
│  输出: List[RetrievalPath]                                   │
│                                                              │
│  q = text_embed(query)          # 查询嵌入 [D]，冻结         │
│  z = q.clone()                  # 初始潜在状态               │
│                                                              │
│  for round in range(max_rounds):  ← ACT halt 控制            │
│  │                                                           │
│  │  ┌─── Phase 1: L1 粗粒度路由 (Cross-Attention) ──┐       │
│  │  │ s1, w1, k1* = selector(q+z, M_L1)             │       │
│  │  │   Q=W_q(q+z), K=W_k(M_L1), V=W_v(M_L1)       │       │
│  │  │   s1 = W_o(MultiHeadAttn(Q,K,V))   # 软选择    │       │
│  │  │   k1* = argmax(mean_head_scores)    # 硬索引    │       │
│  │  │ z = z + s1                                     │       │
│  │  │ for _ in L_cycles:                             │       │
│  │  │   z = L_level(z, s1 + q)                       │       │
│  │  └────────────────────────────────────────────────┘       │
│  │                                                           │
│  │  ┌─── Phase 2: L2 细粒度聚焦 (Cross-Attention) ──┐       │
│  │  │ M_L2 = children_embeds(L1[k1*])                │       │
│  │  │ s2, w2, k2* = selector(q+z, M_L2)             │       │
│  │  │ z = z + s2                                     │       │
│  │  │ for _ in L_cycles:                             │       │
│  │  │   z = L_level(z, s2 + q)                       │       │
│  │  └────────────────────────────────────────────────┘       │
│  │                                                           │
│  │  ┌─── Phase 3: L3 精确定位 (Cross-Attention) ────┐       │
│  │  │ M_L3 = children_embeds(L2[k2*])                │       │
│  │  │ s3, w3, k3* = selector(q+z, M_L3)             │       │
│  │  │ z = z + s3                                     │       │
│  │  └────────────────────────────────────────────────┘       │
│  │                                                           │
│  │  collected.append(Path(k1*, k2*, k3*))                    │
│  │                                                           │
│  │  ┌─── ACT Halt Decision ───────────────────┐             │
│  │  │ halt_logit = q_head(z)                   │             │
│  │  │ if halt_logit > 0 and round_idx > 0:     │             │
│  │  │     break  # 至少跑 1 轮                  │             │
│  │  └──────────────────────────────────────────┘             │
│  │                                                           │
│  └── z 状态保留到下一轮（累积已检索信息）                     │
│                                                              │
│  return collected                                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 算法伪代码

```python
class RecursiveRetriever:
    """TRM 式递归检索器（Cross-Attention 版本）。"""

    def __init__(self, embed_dim: int, num_heads: int, L_layers: int, L_cycles: int, max_rounds: int):
        self.selector = CrossAttentionSelector(embed_dim, num_heads)  # 共享，跨层级
        self.L_level = ReasoningModule(embed_dim, L_layers)           # 共享，跨层级
        self.q_head = Linear(embed_dim, 1)                            # ACT halt head
        self.L_cycles = L_cycles
        self.max_rounds = max_rounds

    def forward(
        self, q: Tensor, tree: TreeIndex, return_internals: bool = False
    ) -> Dict[str, Any]:
        z = q.clone()                   # [B, D]，初始潜在状态 = 查询嵌入
        paths = []

        for round_idx in range(self.max_rounds):
            # ── 三阶段树遍历（一次完整 root-to-leaf） ──
            path, z, step_attns = self._traverse_one_path(q, z, tree)
            paths.append(path)

            # ── ACT halt（推理模式，至少走 1 轮） ──
            halt_logit = self.q_head(z)  # [B, 1]
            if not self.training and halt_logit.item() > 0 and round_idx > 0:
                break

        return {"paths": paths, "num_rounds": len(paths), "z_final": z}

    def _traverse_one_path(self, q, z, tree):
        """单次 root → L1 → L2 → L3 遍历。"""

        # Phase 1
        k1, z = self._select_and_reason(q, z, tree.l1_embeddings())
        # Phase 2
        k2, z = self._select_and_reason(q, z, tree.l2_embeddings_of(k1))
        # Phase 3
        k3, z = self._select_and_reason(q, z, tree.l3_embeddings_of(k1, k2))

        return Path(k1, k2, k3), z

    def _select_and_reason(self, q, z, M):
        """单层: Cross-Attention 选择 + L_cycles 内循环推理。"""

        # Navigate: Cross-Attention
        state = q + z                                      # [B, D]
        selected_info, attn_w, k_star = self.selector(state, M)
        #   Q = W_q(state)   →  [B, 1, D]
        #   K = W_k(M)       →  [B, N, D]
        #   V = W_v(M)       →  [B, N, D]
        #   selected_info = W_o(MultiHeadAttn(Q, K, V))    [B, D]  ← 可微
        #   k_star = argmax(avg_head_scores)                        ← 路径索引

        # Update: z += attention 加权信息
        z = z + selected_info

        # Reason: TRM L-level MLP 内循环
        for _ in range(self.L_cycles):
            z = self.L_level(z, selected_info + q)

        return k_star, z
```

### 5.3 多轮检索的 z 状态流

```
Round 1:                                    Round 2:
z₀ = q                                     z₃ 来自 Round 1（包含已检索信息）
  │                                           │
  ├─ Phase1: CA(q+z₀, M_L1) → s1            ├─ Phase1: CA(q+z₃, M_L1) → s4
  │  z₁ = z₀ + s1 → L_level×L_cycles        │  z₄ = z₃ + s4 → L_level×L_cycles
  ├─ Phase2: CA(q+z₁, M_L2[k1]) → s2       ├─ Phase2: CA(q+z₄, M_L2[k4]) → s5
  │  z₂ = z₁ + s2 → L_level×L_cycles        │  z₅ = z₄ + s5 → L_level×L_cycles
  ├─ Phase3: CA(q+z₂, M_L3[k2]) → s3       ├─ Phase3: CA(q+z₅, M_L3[k5]) → s6
  │  z₃ = z₂ + s3                            │  z₆ = z₅ + s6
  │                                           │
  ACT: q_head(z₃) < 0 → 继续                ACT: q_head(z₆) > 0 → 停止
                                              → 返回 [Path(k1,k2,k3), Path(k4,k5,k6)]

关键: s = W_o(MultiHeadAttn(W_q(q+z), W_k(M), W_v(M)))
      比简单 cosine 路由更强:
        - 学习的投影决定"关注什么"进行选择
        - Multi-head 捕获多种相关性信号
        - V 投影让"更新信息"与"匹配键"解耦
      z₃ 累积了 Round 1 的 attention 信息
        → (q + z₃) 的投影方向自动偏离已选区域
```

---

## 6. ACT Halt 训练方案

### 6.1 训练目标

```
ACT halt head 学习: "已收集的信息是否足以回答 query"

                     ┌──────────────────────────────┐
                     │  reward 定义                   │
                     │                                │
                     │  R = answer_quality - λ·rounds │
                     │                                │
                     │  answer_quality:               │
                     │    文本QA: EM / F1 score        │
                     │    视频QA: 选项匹配准确率       │
                     │                                │
                     │  λ: 步数惩罚系数               │
                     │    鼓励用更少轮次达到同样质量   │
                     └──────────────────────────────┘
```

### 6.2 训练数据

| 数据集 | 模态 | 样本量 | reward 信号 |
|--------|------|--------|-------------|
| LongBench | 文本 | ~5K | ground truth → F1 |
| NarrativeQA | 文本 | ~30K | ground truth → ROUGE |
| VideoMME | 视频 | ~2K | 选项匹配 → 0/1 |

### 6.3 训练流程

```python
# 训练伪代码
for query, tree, ground_truth in dataset:
    q = text_embed(query)
    z = q.copy()
    total_loss = 0

    for round_idx in range(max_rounds):
        path, z = retriever.traverse_one_path(q, z, tree)

        # halt 决策
        halt_logit = q_head(z)

        # 用当前已收集的 context 生成答案，计算质量
        context = collect_raw_content(paths_so_far)
        answer = generator(query, context)
        quality = compute_score(answer, ground_truth)  # EM/F1

        # Q-learning target
        # 如果现在停 → reward = quality
        # 如果继续   → reward = γ · future_quality - λ
        target_q = quality if is_last else γ * next_quality - λ
        loss += mse(sigmoid(halt_logit), target_q)

    total_loss.backward()
    optimizer.step()
```

### 6.4 可训练组件 vs 冻结组件

```
冻结 (不训练):
  ✗ text_embed()        # 预训练嵌入器，冻结
  ✗ TreeIndex embeddings # 预计算的节点嵌入，冻结

可训练:
  ✓ CrossAttentionSelector (W_q, W_k, W_v, W_o)  # 节点选择投影
  ✓ L_level (ReasoningModule: RMSNorm + SwiGLU)   # MLP 推理模块
  ✓ q_head (ACT halt)                              # 停止决策头
```

---

## 7. 检索结果与答案生成的接口

```python
@dataclass
class RetrievalPath:
    """一条 root-to-leaf 路径。"""
    k1: int                    # L1 索引
    k2: int                    # L2 索引
    k3: int                    # L3 索引
    l1_summary: str            # L1 摘要
    l2_description: str        # L2 描述
    l3_description: str        # L3 描述
    raw_content: Optional[str] # 原始文本（文本模式）
    frame_path: Optional[str]  # 帧路径（视频模式）
    timestamp: Optional[float] # 时间戳（视频模式）


@dataclass
class RetrievalResult:
    """检索器输出。"""
    query: str
    paths: List[RetrievalPath]  # 多轮收集的路径
    num_rounds: int             # 实际检索轮次
    z_final: ndarray            # 最终潜在状态


# 答案生成器接口
def generate_answer(query: str, result: RetrievalResult) -> str:
    if is_text_mode(result):
        context = "\n".join(p.raw_content for p in result.paths)
        return LLM(query=query, context=context)
    else:
        frames = [load_image(p.frame_path) for p in result.paths]
        captions = [p.l3_description for p in result.paths]
        return VLM(query=query, images=frames, captions=captions)
```

---

## 8. 与现有系统的关系

```
参考代码                            新架构                    变化
─────────────────────────────────────────────────────────────────
video_pyramid.py (HSP)            → TreeIndex               重构为统一格式
video_tree_trm.py (cosine路由)    → RecursiveRetriever      Cross-Attention+ACT
  select_next_node (CrossAttn)    → CrossAttentionSelector   保留CA思路，简化为向量级
  L_level (Transformer blocks)    → ReasoningModule          MLP-based (向量非序列)
visual_projection.py              → 删除                    L3 全文本化
video_indexer.py (CLIP encode)    → embeddings.py            统一 text_embed()
pipeline.py                       → pipeline.py             ✅ 已实现（含延迟 embed 策略）
answer_generator.py               → answer_generator.py     ✅ 已实现
config.py                         → config.py               全面重构

新增:
  + tree_index.py                 统一数据结构              ✅ 已实现
  + embeddings.py                 嵌入服务封装              ✅ 已实现
  + llm_client.py                 LLM/VLM 客户端            ✅ 已实现
  + text_tree_builder.py          文本模式预处理            ✅ 已实现
  + video_tree_builder.py         视频模式预处理            ✅ 已实现
  + recursive_retriever.py        TRM 递归检索器 (CA+MLP+ACT) ✅ 已实现
  + losses.py                     NavigationLoss + ACTLoss  ✅ 已实现
  + train.py                      两阶段训练入口            ✅ 已实现
  + main.py                       推理/演示入口             ✅ 已实现
```

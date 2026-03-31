# Video-Tree-TRM 训练方案（Phase 1 导航 + Phase 2 ACT）

> **目标**: 使用已建好的 218 个视频树，按照 80% 训练集 / 20% 验证集划分，执行两阶段训练。
>
> **核心思想**: 利用 CrossAttention 计算视频树 L1→L2→L3 路径与「问题+正确答案」的相关度，训练模型找到最正确的帧。

---

## 1. 数据集划分

### 1.1 划分策略

| 集合 | 视频数 | 样本数（估算） | 用途 |
|------|--------|----------------|------|
| 训练集 | 218 × 80% ≈ **174** | ~520 | Phase 1 + Phase 2 训练 |
| 验证集 | 218 × 20% ≈ **44** | ~130 | 训练过程监控、早停 |

### 1.2 执行脚本

```bash
# 重新划分：80% train / 20% val（无 test）
python scripts/split_videomme_dataset.py \
    --input data/videomme/queries/sample_eval.jsonl \
    --output-dir data/videomme/splits \
    --stage 1 \
    --train-ratio 0.8
```

### 1.3 输出文件

```
data/videomme/splits/
├── train.jsonl              # 训练集（~520 样本，174 视频）
├── split_manifest.json      # 元信息（含 train_youtube_ids）
```

---

## 2. Ground Truth 路径推导

### 2.1 核心逻辑

对于 VideoMME 数据集，每个 QA 样本包含：
- `query`: 问题文本
- `answer`: 正确选项字母（A/B/C/D）
- `options`: 4 个选项文本
- `timestamp`: 0.0（VideoMME 无精确时间戳）

**GT 路径推导策略**：

```
1. 从 answer 提取正确选项的完整文本
   answer="B" + options=["A. xxx", "B. yyy", ...] → correct_option="yyy"

2. 遍历视频树所有 L3 节点
   计算 token_f1(L3.description, correct_option)
   返回 F1 分数最高的路径 (l1_idx, l2_idx, l3_idx)
```

### 2.2 代码实现（已存在于 train.py）

```python
def find_gt_path_video(tree, timestamp, correct_option):
    # timestamp=0 时，退化为文本匹配
    if correct_option:
        best_score, best_path = -1.0, None
        for i, l1 in enumerate(tree.roots):
            for j, l2 in enumerate(l1.children):
                for k, l3 in enumerate(l2.children):
                    score = token_f1(l3.description, correct_option)
                    if score > best_score:
                        best_score = score
                        best_path = (i, j, k)
        return best_path
    return None
```

### 2.3 为什么用「正确答案」而非「问题」

- **问题**: 多个选项可能都包含问题中的关键词，无法区分
- **正确答案**: 包含该帧独有的视觉信息，能精确定位到包含答案内容的帧

**示例**：
```
问题: "视频中发生了什么事？"
选项: ["A. 切洋葱", "B. 煮面条", "C. 装盘", "D. 调味"]
答案: B

正确答案文本 "煮面条" 与 L3 描述 "厨师将面条放入沸水中" 高度相关
→ 模型学会通过答案内容定位到正确帧
```

---

## 3. Phase 1：导航训练

### 3.1 训练目标

训练 `CrossAttentionSelector` + `ReasoningModule` 正确导航到 GT 路径。

### 3.2 损失函数

**NavigationLoss**（三层 Cross-Entropy）：

```
L_nav = (1/3) × Σ_{level∈{L1,L2,L3}} -log(attn_weight[level][gt_idx])
```

- `attn_weight[level]`: CrossAttention 在该层的 softmax 权重
- `gt_idx`: Ground Truth 节点索引

### 3.3 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_rounds` | **1** | Phase 1 单轮遍历 |
| `max_epochs` | 30 | 训练轮数 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 1e-5 | 权重衰减 |
| `batch_size` | 1 | 批大小（可按显存调整）|
| `eval_interval` | 5 | 每 N epoch 保存检查点 |
| `act_loss_weight` | **0.0** | Phase 1 不使用 ACT 损失 |

### 3.4 训练流程伪代码

```python
# 初始化
retriever = RecursiveRetriever(config)
retriever.max_rounds = 1  # Phase 1: 单轮
optimizer = AdamW(retriever.parameters(), lr=1e-4)
nav_loss_fn = NavigationLoss()

for epoch in range(30):
    for sample in train_dataset:
        q = embed_model(sample["query"])
        gt_path = sample["gt_path"]  # (l1_idx, l2_idx, l3_idx)

        # 前向传播
        result = retriever(q, sample["tree"], return_internals=True)
        # result["attn_weights_per_step"] = [attn_L1, attn_L2, attn_L3]

        # 计算损失
        loss = nav_loss_fn(result["attn_weights_per_step"][:3], gt_path)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.5 验证集评估

每 `eval_interval` epoch 在验证集上评估：

```python
correct = 0
for sample in val_dataset:
    result = retriever(q, sample["tree"])
    pred_path = result["paths"][0]  # 预测路径
    if pred_path == sample["gt_path"]:
        correct += 1
accuracy = correct / len(val_dataset)
```

---

## 4. Phase 2：ACT 训练

### 4.1 训练目标

训练 `q_head` 学习何时停止检索（halt decision）。

### 4.2 损失函数

**ACTLoss**（Q-learning 二分类）：

```
Q_halt(t)    = answer_quality[t]
Q_continue(t) = γ × answer_quality[t+1] - λ

target = 1.0 if Q_halt ≥ Q_continue else 0.0
L_act = BCE(sigmoid(halt_logit[t]), target)
```

**总损失**：

```
L_total = nav_weight × L_nav + act_weight × L_act
```

### 4.3 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_rounds` | **5** | Phase 2 多轮遍历 |
| `max_epochs` | 20 | 训练轮数 |
| `checkpoint` | Phase1_best.pt | 加载 Phase 1 权重 |
| `nav_loss_weight` | 1.0 | 导航损失权重 |
| `act_loss_weight` | 0.1 | ACT 损失权重 |
| `act_lambda_step` | 0.1 | 步数惩罚 |
| `act_gamma` | 0.9 | 折扣因子 |

### 4.4 Answer Quality 计算

```python
# 每轮检索后，用当前收集的帧生成答案
for t in range(result["num_rounds"]):
    paths_so_far = result["paths"][:t+1]
    frames = [tree.get_node(*p).frame_path for p in paths_so_far]
    pred_answer = VLM(query, frames)
    quality[t] = compute_accuracy(pred_answer, gt_answer)
```

### 4.5 训练流程伪代码

```python
# 加载 Phase 1 权重
retriever.load_state_dict(torch.load("phase1_best.pt"))
retriever.max_rounds = 5  # Phase 2: 多轮

for epoch in range(20):
    for sample in train_dataset:
        result = retriever(q, sample["tree"], return_internals=True)

        # 计算每轮答案质量
        qualities = []
        for t in range(result["num_rounds"]):
            pred = generator.generate(query, result["paths"][:t+1], tree)
            qualities.append(token_f1(pred, sample["answer"]))

        # 计算损失
        loss_dict = compute_nav_act_loss(
            result, sample["gt_path"], qualities,
            nav_loss_fn, act_loss_fn,
            nav_weight=1.0, act_weight=0.1
        )

        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()
```

---

## 5. 训练命令

### 5.1 Phase 1 训练

```bash
conda run -n Video-Tree-TRM python train.py \
    --config config/videomme.yaml \
    --set train.dataset_path=data/videomme/splits/train.jsonl \
    --set train.max_epochs_phase1=30 \
    --set train.max_epochs_phase2=0 \
    --set train.save_dir=data/videomme/checkpoints/phase1
```

### 5.2 Phase 2 训练

```bash
conda run -n Video-Tree-TRM python train.py \
    --config config/videomme.yaml \
    --set train.dataset_path=data/videomme/splits/train.jsonl \
    --set retriever.checkpoint=data/videomme/checkpoints/phase1/best.pt \
    --set train.max_epochs_phase1=0 \
    --set train.max_epochs_phase2=20 \
    --set train.save_dir=data/videomme/checkpoints/phase2
```

---

## 6. 评估指标

### 6.1 导航准确率

| 指标 | 计算方式 |
|------|---------|
| **Nav Acc @L1** | 预测的 L1 索引 == GT L1 |
| **Nav Acc @L2** | L1 正确 且 L2 索引 == GT L2 |
| **Nav Acc @L3** | L1,L2 正确 且 L3 索引 == GT L3 |
| **Path Acc** | 完整路径 (L1,L2,L3) 全部正确 |

### 6.2 效率指标

| 指标 | 计算方式 |
|------|---------|
| **Avg Rounds** | 平均检索轮次 |
| **Halt Accuracy** | halt_logit > 0 时实际质量已收敛的比例 |

---

## 7. 预期产出

### 7.1 检查点

```
data/videomme/checkpoints/
├── phase1/
│   ├── phase1_epoch5.pt
│   ├── phase1_epoch10.pt
│   ├── ...
│   └── best.pt          # 验证集 Path Acc 最高的检查点
└── phase2/
    ├── phase2_epoch5.pt
    ├── ...
    └── best.pt
```

### 7.2 训练日志

```
logs/
├── system.log           # 训练过程日志
└── metrics.json         # JSON 格式的指标记录
```

---

## 8. 实施检查清单

### 数据准备
- [ ] 执行 `split_videomme_dataset.py --stage 1 --train-ratio 0.8`
- [ ] 确认 train.jsonl 中的视频树均已构建
- [ ] 统计 GT 路径推导成功率（应 > 50%）

### Phase 1
- [ ] 训练 loss 下降正常
- [ ] 验证集 Nav Acc @L3 > 随机 baseline
- [ ] 保存 best.pt 检查点

### Phase 2
- [ ] 加载 Phase 1 检查点成功
- [ ] ACT loss 收敛
- [ ] Avg Rounds 减少（相比 max_rounds=5）

---

## 9. 与当前实现的差异

| 项目 | 当前实现 | 本方案 |
|------|---------|--------|
| 数据划分 | 80% train / 20% pending | **80% train / 20% val** |
| Phase 1 epochs | 30 | 30 |
| Phase 2 epochs | 20 | 20 |
| 验证集使用 | 未明确 | **每 eval_interval 评估** |
| GT 路径 | 基于 correct_option 文本匹配 | **同左，已正确实现** |

---

## 10. 风险与对策

| 风险 | 对策 |
|------|------|
| GT 路径推导率低 | 调整 token_f1 阈值，或使用 embedding cosine 相似度 |
| 显存不足 | 减小 batch_size，或使用 gradient accumulation |
| 训练不收敛 | 降低 lr，增加 warmup steps |
| ACT 训练不稳定 | 减小 act_loss_weight，先固定 nav 权重 |

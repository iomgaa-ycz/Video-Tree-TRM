# Video-Tree-TRM 实验计划

> **目标**: 验证结合 TRM 多层推理探索能力（Cross-Attention Selector + ACT Halt）
> 与 PageIndex 树状检索能力的 Video-Tree-TRM 在长文本和长视频问答任务上的效果。
>
> **两条并行实验线路**：
> - **线路 A** — 长文本检索（LongBench / NarrativeQA）
> - **线路 B** — 长视频检索（VideoMME）

---

## 目录

- [评测体系](#1-评测体系)
- [实验环境准备](#2-实验环境准备)
- [线路 A：长文本检索](#3-线路-a长文本检索)
- [线路 B：长视频检索](#4-线路-b长视频检索)
- [基线方法](#5-基线方法)
- [消融实验](#6-消融实验)
- [里程碑检查表](#7-里程碑检查表)
- [结果记录表](#8-结果记录表)

---

## 1. 评测体系

### 1.1 质量指标

| 指标 | 适用模态 | 计算方式 |
|------|---------|---------|
| **EM (Exact Match)** | 文本 QA | 标准化后精确字符串匹配，0/1 |
| **F1** | 文本 QA | token 级 precision/recall，`token_f1()` 已实现于 `answer_generator.py` |
| **Accuracy** | 视频 QA（多选） | 模型选项与标准答案选项完全匹配的比率 |

### 1.2 效率指标

| 指标 | 说明 |
|------|------|
| **Avg Rounds** | 所有样本的平均检索轮次（衡量 ACT Halt 效果，越少越好） |
| **Max Rounds Hit Rate** | 触达 `max_rounds` 上限而非主动停止的样本比率 |

### 1.3 诊断指标

| 指标 | 说明 |
|------|------|
| **Nav Accuracy @L1** | 第一轮检索选中正确 L1 节点的比率 |
| **Nav Accuracy @L2** | 在正确 L1 下选中正确 L2 节点的比率 |
| **Nav Accuracy @L3** | 在正确 L2 下选中正确 L3 节点的比率 |

---

## 2. 实验环境准备

### 2.1 安装依赖

```bash
conda activate Video-Tree-TRM
pip install -r requirements.txt
```

### 2.2 配置文件

复制并填写环境变量：

```bash
cp .env.example .env
# 填写 LLM_API_KEY / VLM_API_KEY / EMBED_API_KEY（若使用远程嵌入）
```

默认配置文件 `config/default.yaml` 已预置：

```yaml
embed.model_name: "BAAI/bge-base-zh-v1.5"   # 嵌入模型
retriever.embed_dim: 2560
retriever.max_rounds: 5
train.max_epochs_phase1: 30
train.max_epochs_phase2: 20
```

> 消融实验时通过 `--set key=value` 覆盖，无需修改 yaml 文件。

---

## 3. 线路 A：长文本检索

> **优先级**: P0（LongBench）→ P1（NarrativeQA）
> **并行方式**: 与线路 B 同步推进，互不阻塞

### A.1 数据集准备

| 数据集 | 样本量 | 格式 | 优先级 |
|--------|--------|------|--------|
| **LongBench** | ~5K | JSONL `{"query", "answer", "source_path", "modality": "text"}` | P0 |
| **NarrativeQA** | ~30K | 同上 | P1 |

```bash
# 下载 LongBench（示例）
mkdir -p data/longbench data/narrativeqa
# 将转换后的 JSONL 文件放置于对应目录
```

配置覆盖：

```bash
# LongBench
--set train.dataset=longbench --set train.dataset_path=data/longbench

# NarrativeQA（P1，LongBench 完成后）
--set train.dataset=narrativeqa --set train.dataset_path=data/narrativeqa
```

### A.2 索引构建

```bash
# 对所有文本文件批量构建 TreeIndex（含磁盘缓存）
conda run -n Video-Tree-TRM python main.py index \
    --source data/longbench/doc.txt \
    --modality text \
    --config config/default.yaml
```

> 缓存命中后重复运行零开销，构建结果保存至 `cache/trees/`。

### A.3 两阶段训练

```bash
# Phase 1：导航训练（单轮，~30 epoch）
conda run -n Video-Tree-TRM python train.py \
    --config config/default.yaml \
    --set train.dataset=longbench \
    --set train.dataset_path=data/longbench

# Phase 2 权重加载自 Phase 1 最佳检查点
# 在 config/default.yaml 中设置:
#   retriever.checkpoint: checkpoints/phase1_epoch30.pt
#   train.max_epochs_phase2: 20
conda run -n Video-Tree-TRM python train.py \
    --config config/default.yaml \
    --set retriever.checkpoint=checkpoints/phase1_epoch30.pt
```

训练检查点保存至 `checkpoints/phase1_epoch{N}.pt` / `phase2_epoch{N}.pt`。

### A.4 推理评测

```bash
# 单样本问答验证
conda run -n Video-Tree-TRM python main.py query \
    --source data/longbench/doc.txt \
    --modality text \
    --question "文档的核心观点是什么？" \
    --config config/default.yaml
```

批量评测（自行实现评测脚本）：
- 遍历测试集 JSONL
- 调用 `Pipeline.query()` 获取预测答案
- 计算 EM / F1，汇总 Avg Rounds

### A.5 基线对比

见 [第 5 节](#5-基线方法)。

---

## 4. 线路 B：长视频检索

> **优先级**: P2（可与线路 A 并行推进）

### B.1 数据集准备

| 数据集 | 样本量 | 格式 | 评测指标 |
|--------|--------|------|---------|
| **VideoMME** | ~2K | JSONL `{"query", "answer", "source_path", "modality": "video", "timestamp": float}` | Accuracy（多选） |

```bash
mkdir -p data/videomme
# 下载 VideoMME 并转换为上述 JSONL 格式
```

### B.2 索引构建

```bash
conda run -n Video-Tree-TRM python main.py index \
    --source data/videomme/video.mp4 \
    --modality video \
    --config config/default.yaml
```

关键配置：

```yaml
tree:
  l1_segment_duration: 600.0   # L1 段时长（秒）
  l2_clip_duration: 60.0       # L2 clip 时长（秒）
  l3_fps: 1.0                  # L3 帧提取频率
  l2_representative_frames: 10 # VLM 描述用代表帧数
```

### B.3 两阶段训练

```bash
# Phase 1（视频模态）
conda run -n Video-Tree-TRM python train.py \
    --config config/default.yaml \
    --set train.dataset=videomme \
    --set train.dataset_path=data/videomme

# Phase 2（视频模态，加载 Phase 1 检查点）
conda run -n Video-Tree-TRM python train.py \
    --config config/default.yaml \
    --set train.dataset=videomme \
    --set train.dataset_path=data/videomme \
    --set retriever.checkpoint=checkpoints/phase1_epoch30.pt
```

### B.4 推理评测

```bash
conda run -n Video-Tree-TRM python main.py query \
    --source data/videomme/video.mp4 \
    --modality video \
    --question "视频中发生了什么事？" \
    --config config/default.yaml
```

批量评测：遍历测试集 → `Pipeline.query()` → 与多选选项比对 → 计算 Accuracy。

### B.5 基线对比

见 [第 5 节](#5-基线方法)。

---

## 5. 基线方法

| 方法 | 描述 | 实现方式 |
|------|------|---------|
| **BM25 + LLM** | 传统稀疏检索 | `rank_bm25` 库检索 top-k 段落 → 拼接上下文 → `LLMClient.chat()` |
| **Dense Retrieval + LLM** | BGE 向量检索 + rerank | `EmbeddingModel.embed_tensor()` 全量检索 top-k → rerank → 生成 |
| **PageIndex（无 TRM）** | 树状导航，cosine 路由，无推理模块 | 替换 `CrossAttentionSelector` 为 cosine 相似度选节点 |
| **Tree-TRM（原论文）** | 原始实现 | 参考 `Reference/Tree-TRM/` 目录 |
| **Video-Tree-TRM（ours）** | 本项目实现 | `Pipeline.query()` |

评测时各方法使用相同数据集和评测脚本，确保公平对比。

---

## 6. 消融实验

在主实验（LongBench）最优配置基础上，逐一变更单一变量：

| 编号 | 变量 | 候选值 | 配置覆盖 | 预期观察 |
|------|------|--------|---------|---------|
| **A1** | 选择器类型 | Cross-Attention vs Cosine | 替换 `CrossAttentionSelector` 实现 | CA 路由是否带来 F1 提升 |
| **A2** | 推理深度 | L_cycles ∈ {1, 2, 4, 8} | `--set retriever.L_cycles=N` | 质量-计算量权衡 |
| **A3** | 推理模块层数 | L_layers ∈ {1, 2, 4} | `--set retriever.L_layers=N` | 网络深度的边际收益 |
| **A4** | 多轮检索上限 | max_rounds ∈ {1, 3, 5} | `--set retriever.max_rounds=N` | ACT 多轮边际收益 |
| **A5** | ACT Halt 机制 | 有 / 无 | `act_loss_weight=0.0` 禁用 | ACT 对效率和质量的贡献 |
| **A6** | 注意力头数 | num_heads ∈ {1, 4, 8} | `--set retriever.num_heads=N` | 多头注意力的容量影响 |

每组消融实验保持其余超参数为默认值（`config/default.yaml`）。

---

## 7. 里程碑检查表

### 线路 A（文本）

- [ ] 环境配置完成，`python main.py --help` 正常输出
- [ ] LongBench 数据集下载 & 转换为 JSONL 格式
- [ ] LongBench 索引构建完成（`cache/trees/` 缓存生成）
- [ ] Phase 1 训练完成（文本，LongBench）
- [ ] Phase 2 训练完成（文本，LongBench）
- [ ] LongBench 批量评测完成（EM / F1 / Avg Rounds）
- [ ] 4 条基线方法评测完成
- [ ] NarrativeQA 实验（P1，可选）

### 线路 B（视频）

- [ ] VideoMME 数据集下载 & 转换为 JSONL 格式
- [ ] VideoMME 索引构建完成（视频帧提取 + VLM 描述生成）
- [ ] Phase 1 训练完成（视频，VideoMME）
- [ ] Phase 2 训练完成（视频，VideoMME）
- [ ] VideoMME 批量评测完成（Accuracy / Avg Rounds）

### 消融实验

- [ ] A1: Cross-Attention vs Cosine 路由
- [ ] A2: L_cycles 扫描（1/2/4/8）
- [ ] A3: L_layers 扫描（1/2/4）
- [ ] A4: max_rounds 扫描（1/3/5）
- [ ] A5: 有/无 ACT Halt
- [ ] A6: num_heads 扫描（1/4/8）

---

## 8. 结果记录表

### 8.1 主实验结果

| 方法 | LongBench EM | LongBench F1 | VideoMME Acc | Avg Rounds |
|------|-------------|-------------|-------------|-----------|
| BM25 + LLM | | | | — |
| Dense Retrieval + LLM | | | | — |
| PageIndex（无 TRM） | | | | |
| Tree-TRM（原论文） | | | | |
| **Video-Tree-TRM（ours）** | | | | |

### 8.2 消融实验结果（LongBench F1）

| 变量 | 值 | F1 | Avg Rounds | 备注 |
|------|----|----|-----------|------|
| 选择器 | Cross-Attention | | | 默认 |
| 选择器 | Cosine | | | A1 |
| L_cycles | 1 | | | A2 |
| L_cycles | 2 | | | A2 |
| L_cycles | 4 | | | A2 默认 |
| L_cycles | 8 | | | A2 |
| L_layers | 1 | | | A3 |
| L_layers | 2 | | | A3 默认 |
| L_layers | 4 | | | A3 |
| max_rounds | 1 | | | A4 |
| max_rounds | 3 | | | A4 |
| max_rounds | 5 | | | A4 默认 |
| ACT | 有 | | | A5 默认 |
| ACT | 无 | | | A5 |
| num_heads | 1 | | | A6 |
| num_heads | 4 | | | A6 默认 |
| num_heads | 8 | | | A6 |

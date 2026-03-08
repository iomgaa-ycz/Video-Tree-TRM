# CLAUDE.md

> [!URGENT]
> **研究性项目 (Research Project)**
> 1. 本项目为 MVP（最小可行性产品），严禁过度工程化。
> 2. 你的所有思考过程和回复必须使用 **简体中文**。

## 1. 项目元数据 (Metadata)
- **核心目标**: 构建结合TRM多层训话探索能力与PageIndex树状检索能力的新型Video RAG
- **项目类型**: MVP / 研究性项目
- **后端架构**: Python 3.11
- **版本管理**: Git
- **Conda 环境**: Video-Tree-TRM (Python 3.11)

## 2. 常用命令 (Commands)

### 2.1 Conda 环境管理

> [!CRITICAL]
> **所有 Python 相关命令必须在 Video-Tree-TRM 环境中执行**
> - 使用 `conda run -n Video-Tree-TRM <command>` 确保命令在正确环境中运行
> - 或在命令前显式添加 `source activate Video-Tree-TRM &&`
> - 如果需要使用llm可以依据 '.env'环境变量文件使用

```bash
# 激活项目环境（交互式 shell）
conda activate Video-Tree-TRM

# 推荐：使用 conda run 执行命令（自动使用正确环境）
conda run -n Video-Tree-TRM pip install xxx
conda run -n Video-Tree-TRM python -m pytest xxx #注意不能conda run -n Video-Tree-TRM pytest xxx，因为这样子pytest不会调用Video-Tree-TRM
conda run -n Video-Tree-TRM python xxx

# 或者：在命令前激活环境
source activate Video-Tree-TRM && xxx
```

### 2.2 代码质量检查
```bash
# 代码格式化
conda run -n Video-Tree-TRM ruff format xxx

# 代码检查并自动修复
conda run -n Video-Tree-TRM ruff check xxx --fix
```


## 3. 标准作业程序 (Standard Operating Procedure)
> **Agent 必须严格遵守以下生命周期执行任务：**

### Phase 1: 规划与设计 (Planning)
1. **查阅规格 (Read Specs)&讨论**: 在撰写计划前，**必须**仔细阅读 `docs/` 下对应的文档与`.report`下的项目整理架构，并使用GitNexus MCP来了解整个项目的最新情况。对于不理解的地方请与人类进行多轮讨论，确保理解人类的设计意图。
2. **计划 (Plan)**: 正式编码前，**必须**使用plan模式输出开发计划，内容必须严格包含：
   - **1.1 摘要 (Summary)**: 1-2句话的简单总结。
   - **1.2 审查点 (User Review Required)**: 明确列出整个计划中不清楚、需要用户审查和确认的部分。若无，请注明"无"。
   - **1.3 拟议变更 (Proposed Changes)**:
     - 以 **文件名 + 修改内容** 的形式列出。
     - 修改内容必须精确到 **函数/方法级别 (Function-level)**。
     - 明确标识 `[NEW]`, `[MODIFY]`, `[DELETE]`。
   - **1.4 验证计划 (Verification Plan)**: 具体描述如何验证修改是否成功（如具体的测试命令、预期日志输出等）。
4. **等待 (Wait)**: **必须** 暂停并等待用户审核开发计划。用户批准后方可进入下一阶段。

### Phase 2: 执行与验证 (Execution & Verification)
1. **编码 (Coding)**: 审核通过后，开始编写代码。
2. **验证 (Verify)**:
   - **环境检查**: 确保所有命令在 Video-Tree-TRM 环境中执行（使用 `conda run -n Video-Tree-TRM`）
   - **运行验证命令**:
     - *失败*: 回到编码阶段修复，直到通过。
     - *成功*: 进入下一步。

## 4. 核心规则 (Rules)

### 4.1 代码开发规范 (Code Style)
- **类型系统**: 强制所有函数签名包含完整类型注解 (`Union`, `Dict`, `Optional` 等)。
- **文档**: 所有模块、类、方法必须包含 **中文 Docstring** (功能、参数、返回值、关键实现细节)。
- **MVP原则**:
  - **必须** 必须在`tests/`目录下编写测试代码。
  - **严禁** 使用默认参数掩盖仅需逻辑（必须显式传递关键参数）。
  - **必需** 运行时检查：关键维度、设备一致性必须通过 assertion 或 if 验证。
- **代码组织**:
  - 使用阶段化注释 (`# Phase 1`, `# Phase 2`) 组织复杂逻辑。
  - 接口返回值需包含完整诊断信息（输出、损失、统计），使用条件标志控制。
- **命名与依赖**:
  - 类名 `PascalCase`，变量描述性命名，私有变量前缀 `_`。
  - 导入顺序：标准库 → 第三方库 → 项目内部。
- **日志与错误处理**: 使用 `utils/logger_system.py` 的 `log_msg()`, `log_json()`, `ensure()`, `log_exception()`
  - 禁用 `print()`，`log_msg("ERROR")` 不自动抛出异常，输出到 `logs/system.log` + `logs/metrics.json`
- **功能修改**:
  - **必须** 不考虑向后兼容，直接修改原文件。代码简洁性优先。

### 4.2 配置管理规范
- **优先级**: CLI args > `.env` > YAML，三者统一归口到 dataclass
- **文件**: `config/default.yaml`（全量非敏感配置，必须写全）, `.env`（敏感信息，不提交）, `.env.example`（模板）

### 4.3 测试组织规范
- **目录**: `tests/{unit,integration,e2e}/test_*.py`，最低覆盖率 80%
- **运行**: `conda run -n Video-Tree-TRM pytest tests/unit/ --cov=utils --cov-report=term-missing`

#### Agent 测试输出规范

> **除了使用pytest进行单元测试，还必须构建一个完善的main.py文件将所有过程串联起来，并必须将完整执行过程保存为 Markdown 文件，供 Claude Code 智能分析，以消除格式输出正确但是逻辑错误的误差或者实际输出质量低低问题。**

| 要素 | 规范 |
|------|------|
| **输出位置** | `tests/outputs/<test_module>/<test_name>_<timestamp>.md` |
| **触发时机** | 所有涉及 Agent 执行的测试 |
| **内容要求** | 任务描述、每步 Agent 输入/输出/推理过程、工具调用、最终结果 |
| **格式要求** | 结构化 Markdown（标题、代码块、列表），人类可读 |
| **分析方式** | Claude Code 读取 MD 文件，评估推理质量、任务完成度、代码正确性 |

**示例结构**:
```markdown
# Agent 测试: <test_name>
## 任务: <task>
## Step 1: <AgentName>
- 输入: ...
- 输出: ...
- 推理: ...
## Step 2: ...
## 最终结果: ...
```

**pytest 集成**: 使用 fixture 或工具类自动保存，测试结束后输出文件路径。


## 5. 上下文获取与迷途指南 (Context & Navigation)

| 需求 | 文档路径 | 说明 |
|------|----------|------|
| 项目目标与背景 | `README.md` | 核心业务逻辑与项目定性 |
| 架构与模块设计 | `.report/CODEMAPS/{architecture,backend,data}.md` | 整体架构、分层设计、模块依赖 |
| 该项目最主要的参考项目 | `Reference/Tree-TRM`| 特定模块的详细设计 |
| 其他参考项目 | `Reference/PageIndex_ Next-Generation Vectorless, Reasoning-based RAG.md`和 `Reference/Tree-TRM`|  |

## 6. 输出规范

### 6.1 语言要求
- 所有输出语言: **中文**

### 6.2 信息密度原则
- **优先使用**:
  - 简洁文本描述
  - 伪代码（而非完整代码）
  - 表格（对比、配置、参数说明）
  - 流程图（Mermaid）
  - 项目符号列表
- **避免使用**:
  - 大段完整代码（信息密度低，可读性差）
  - 冗长的自然语言解释
- **核心原则**: 用最少的字符传递最多的信息
`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Video-Tree-TRM** (860 symbols, 2247 relationships, 70 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/Video-Tree-TRM/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Video-Tree-TRM/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Video-Tree-TRM/clusters` | All functional areas |
| `gitnexus://repo/Video-Tree-TRM/processes` | All execution flows |
| `gitnexus://repo/Video-Tree-TRM/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->

# 开发者文档问答助手（RAG + 混合检索 + 工具调用 Agent）

基于开源项目官方文档（示例用 FastAPI 文档）构建的开发者问答系统。
当前已实现 **RAG 问答、BM25 + 向量混合检索（RRF 融合）和工具调用 Agent**，并用
自建评测集量化检索与 Agent 工具选择效果。

## 架构

```
用户提问
   │
   ├─► POST /ask ─► BM25 + 向量检索 ─► RRF 融合 ─► 拼上下文 ─► DeepSeek 作答
   │                         ▲
   │                         │
   └─► POST /agent/ask ─► run_agent 工具调用循环 ─► DeepSeek 选择工具 ─┐
                              ▲                                     │
                              ├─ search_docs ─► 复用混合检索 ────────┤
                              └─ lookup_api ─► inspect FastAPI API ──┘
                                     工具结果回传模型，循环至最终回答 + trace

文档(docs/) ─► 切分(chunk) ─► 本地 embedding(Ollama) ─► Chroma
```

- **后端**: FastAPI
- **对话生成**: DeepSeek API（OpenAI 兼容）
- **embedding**: Ollama 本地模型 `nomic-embed-text`（免费、离线，成本与 chat 分离）
- **向量库**: Chroma（持久化到 `data/chroma`）
- **Agent**: OpenAI 兼容 function calling；`search_docs` 用于概念/教程问题，`lookup_api`
  用于查询 FastAPI 公开函数、类和方法的签名
- **引用溯源**: 回答标注 `[来源: 文件名#片段号]`；通过 system prompt 约束模型
  仅依据检索片段作答、无依据时拒答，降低无依据回答（注：暂未做引用与检索来源的一致性校验）

## 快速开始

```bash
# 1. 装依赖（已在 .venv）
./.venv/bin/pip install -r requirements.txt

# 2. 确保 Ollama 在跑，并已拉 embedding 模型
ollama pull nomic-embed-text

# 3. 同步官方文档（钉定 commit，可复现；含示例代码内联）
./.venv/bin/python -m scripts.sync_docs

# 4. 配置 DeepSeek key
cp .env.example .env   # 然后把 DEEPSEEK_API_KEY 填进去

# 5. 手动演示检索（不需要 DeepSeek key）
./.venv/bin/python -m scripts.demo_retrieval

# 6. 启动服务
./.venv/bin/uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000/docs 交互测试
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | /health | 健康检查 + 当前库内片段数 |
| POST | /ingest | 导入 `docs/` 下的 .md/.txt/.pdf |
| POST | /ask    | 提问，返回答案 + 引用来源 |
| POST | /agent/ask | Agent 按需调用 `search_docs` / `lookup_api`，返回答案、steps 和 trace |

## 数据集

- **语料**：FastAPI 官方 tutorial **32 篇**（钉定 commit，`scripts/sync_docs.py` 可复现，
  含 `docs_src` 示例代码内联）。溯源见根目录 `SOURCES.md`、许可见 `LICENSE-FASTAPI.txt`（MIT）。
- **评测集**：`eval/` 下三份 JSONL，共 52 题
  - `dev.jsonl`（24）：**开发集**，允许据此调参
  - `test.jsonl`（16）：**冻结的独立测试集**，仅里程碑评估，不得据此调参
  - `unanswerable.jsonl`（12）：知识库中**没有答案**的问题（off-topic + 貌似相关但未收录）
- 每条标注 `expected_source` 与 `evidence_quote`，问题按 semantic/task/keyword/error **四类严格均衡**
  （dev 每类 6、test 每类 4），使 by-type 指标有意义。

## 检索评测

报告两套**互相独立**的指标（定义见 `app/eval_metrics.py`）：

- **source**：按 source 首次出现去重后的名次——「正确文档整体排第几」（不看证据）。
- **evidence**：不去重，第一个「来源正确且含 `evidence_quote`」的 chunk 名次——「真正能回答的证据片段排第几」，更严格。

下表用**干净重建索引**（`--rebuild-index`）跑出，本地 Ollama embedding，top_n=10：

| 集合 | 检索 | src@1 | src@3 | srcMRR | ev@1 | ev@3 | evMRR |
|------|------|-------|-------|--------|------|------|-------|
| dev  | 纯向量 | 79% | 92% | 0.856 | 54% | 71% | 0.628 |
| dev  | **混合** | **83%** | 92% | **0.894** | 54% | 71% | **0.666** |
| test | 纯向量 | 56% | 100% | 0.771 | 31% | 62% | 0.480 |
| test | **混合** | **88%** | 100% | **0.938** | **50%** | **75%** | **0.621** |

混合检索在两套指标上都 ≥ 纯向量，且在更难的 test 集上提升显著（src@1 +32、ev@1 +19 个点）。
evidence 指标明显低于 source，说明「检索到正确文档」不等于「检索到含答案的片段」——这正是该指标存在的意义。

### Dev-only 优化实验（未接入生产）

以下结果只来自 24 题开发集，用于选择后续方向，**不是 test 泛化结论**。检索层实验衡量
证据片段是否进入上下文，不等同于最终回答质量：

| dev 配置 | evidence 命中 | ev-MRR | source 命中 | 平均上下文 token（估算） |
|----------|---------------|--------|-------------|--------------------------|
| hybrid@4 | 19/24（79.2%） | 0.639 | 23/24（95.8%） | 779 |
| hybrid@6 | 21/24（87.5%） | 0.654 | 24/24（100%） | 1163 |
| DeepSeek rerank + N-win | 23/24（95.8%） | 0.666 | 24/24（100%） | 1699 |

rerank + N-win 在 dev 上把 evidence 命中从 hybrid@6 的 21/24 提高到 23/24，但每题额外需要
一次模型调用；历史缓存记录的平均开销约为 6894 输入 token、849 输出 token 和 5.28 秒延迟。
该结果来自旧 cache key 下的一次有效 dev 运行，当前加固后的 cache-only 模式不能复算，完整限制见
`eval/reports/rerank_dev_report.md`。因此它仅保留为实验工具，**没有接入 `/ask` 默认链路**。

随后用两阶段人工评审比较同一 hybrid 排名的 k4 与 k6 最终回答：第一阶段盲评
correctness/citation，锁定评分后再根据各自真实上下文评 faithfulness。结果如下：

| dev 配置 | overall pass | 总生成 token | 配对胜场 |
|----------|--------------|--------------|----------|
| hybrid@4 | 22/24（91.7%） | 25,068 | 0 |
| hybrid@6 | 22/24（91.7%） | 34,895 | 0 |

k6 的总 token 比 k4 增加 **39.2%**，双方配对胜场均为 0（k6 净胜为 0），
未达到预先冻结的质量门槛。
因此当前结论是**继续保留 hybrid@4 为默认值**；k6 和 rerank 均未因这些实验自动启用为生产配置。
评审流程与防泄漏约束见 `eval/reports/generation_ab_protocol.md`。

```bash
./.venv/bin/python -m scripts.evaluate                               # dev（默认，可据此调参）
./.venv/bin/python -m scripts.evaluate --by-type                     # 按问题类型拆分
./.venv/bin/python -m scripts.evaluate --split test --rebuild-index  # 里程碑：干净索引，仅报告
```

> 里程碑评测务必加 `--rebuild-index`：reset Chroma 后从当前 `docs/` 重新入库再评，避免旧库污染数字。

**不可答问题**（需 DeepSeek key，测系统是否会乱答）：调用问答链路，把原始回答存到
`eval/outputs/` 供**人工复核**（启发式拒答率仅参考，不作最终判定）：

```bash
./.venv/bin/python -m scripts.evaluate_answers
```

## Agent 评测

`eval/agent_dev.jsonl` 是 20 题中文开发集：8 道概念/用法题期望调用 `search_docs`，6 道
API 签名题期望调用 `lookup_api`，3 道混合题期望同时调用两个工具，另有 3 道无关题期望
不调用工具或拒答。评测指标定义如下：

- **Top-1 / 来源命中**：只统计 8 道概念题；期望 source 是否排在第一次 `search_docs`
  返回结果首位 / 是否出现在返回结果中。
- **必需工具召回**：每题标注的期望工具是否全部被调用。
- **多余调用率**：是否调用了标注之外的工具。
- **平均 steps**：每题实际执行工具的平均次数。

在查看结果前，预先设定采纳门槛：英文检索词变体的 Top-1 均值须比基线至少多 **2/8**，
且来源命中率与必需工具召回均不得下降。DeepSeek `deepseek-chat` 每组独立运行 3 次的均值如下：

| Agent dev 配置 | Top-1 | 来源命中 | 必需工具召回 | 多余调用率 | 平均 steps |
|----------------|-------|----------|----------------|--------------|------------|
| 基线（原工具说明） | 3.33/8（41.7%） | 7.67/8（95.8%） | 100% | 13.3% | 1.45 |
| **英文检索词（默认）** | **7.33/8（91.7%）** | **8/8（100%）** | **100%** | **11.7%** | **1.43** |

英文变体 Top-1 平均提升 4 题，并满足其余预设条件，因此已设为默认；仍可通过
`run_agent(..., english_search_query=False)` 关闭。该结论仅来自 **dev-only 小实验**：概念类只有
8 题、每组仅 3 次运行，尚未在冻结 test 集验证，不能视为泛化结论；现有的多余工具调用问题
也尚未专门优化。完整三次原始值见 `eval/reports/agent_english_query_dev.md`。

## 测试

用假 embedding + 临时向量库，不依赖 Ollama / DeepSeek：

```bash
./.venv/bin/python -m pytest -q
```

覆盖：切分/BM25 过滤/幂等入库/source 去冲突/混合检索、**两套指标纯函数**
（source_rank vs evidence_chunk_rank 的独立性回归）、**评测集完整性**
（证据是文档子串、id 唯一、dev/test 不重叠、四类均衡）、**语料无残留 include/tab 标记**，
以及 rerank 安全闸门、缓存校验、两阶段生成 A/B 评审流程、Agent 工具循环/API 与评测指标。
共 200 项，全部离线。

## Roadmap（简历深度点，按顺序做）

- [x] **P0** 混合检索：BM25 + 向量（RRF 融合），提升召回
- [x] **P0** 评测集：真实问题标注正确来源 + 证据，测 recall@1/3 与 MRR
- [x] **P0** 扩充到官方 32 篇文档、52 条评测题（dev/test/unanswerable），数字更有说服力
- [x] **P1（dev-only 实验）** rerank：验证候选重排可提升 evidence 命中；未接入生产默认链路
- [ ] **P1** 父子分块：小块检索、大块喂给模型
- [ ] **P1** Redis：缓存高频 query 的答案
- [x] **P2** Agent 工具调用：按问题调用文档检索或 FastAPI API 签名查询
- [ ] **P2** MySQL：存用户问答记录；Docker 一键部署

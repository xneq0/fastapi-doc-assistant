# 开发者文档问答助手（RAG 问答 + 混合检索）

基于开源项目官方文档（示例用 FastAPI 文档）构建的开发者问答系统。
针对朴素检索召回不准的问题，引入 BM25+向量混合检索，并用自建评测集量化优化效果。

> 说明：当前已实现 **RAG 问答 + 混合检索**。基于工具调用的 **Agent 路由**（区分
> 知识问答 / API 查询）仍在 Roadmap，尚未实现，本文不将其列为已完成能力。

## 架构

```
用户提问
   │
   ▼
FastAPI (/ask)  ──►  检索(Chroma) ──► 拼上下文 ──► DeepSeek 作答(带出处)
   ▲                     ▲
   │                     │ 向量
文档(docs/) ─► 切分(chunk) ─► 本地 embedding(Ollama)
```

- **后端**: FastAPI
- **对话生成**: DeepSeek API（OpenAI 兼容）
- **embedding**: Ollama 本地模型 `nomic-embed-text`（免费、离线，成本与 chat 分离）
- **向量库**: Chroma（持久化到 `data/chroma`）
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

## 测试

用假 embedding + 临时向量库，不依赖 Ollama / DeepSeek：

```bash
./.venv/bin/python -m pytest -q
```

覆盖：切分/BM25 过滤/幂等入库/source 去冲突/混合检索、**两套指标纯函数**
（source_rank vs evidence_chunk_rank 的独立性回归）、**评测集完整性**
（证据是文档子串、id 唯一、dev/test 不重叠、四类均衡）、**语料无残留 include/tab 标记**。
共 101 项，全部离线。

## Roadmap（简历深度点，按顺序做）

- [x] **P0** 混合检索：BM25 + 向量（RRF 融合），提升召回
- [x] **P0** 评测集：真实问题标注正确来源 + 证据，测 recall@1/3 与 MRR
- [x] **P0** 扩充到官方 32 篇文档、52 条评测题（dev/test/unanswerable），数字更有说服力
- [ ] **P1** rerank：对候选片段重排，对比命中率提升
- [ ] **P1** 父子分块：小块检索、大块喂给模型
- [ ] **P1** Redis：缓存高频 query 的答案
- [ ] **P2** Agent 路由：区分「知识问答 / API 查询」走不同路径
- [ ] **P2** MySQL：存用户问答记录；Docker 一键部署

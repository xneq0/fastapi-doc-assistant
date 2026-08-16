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

# 3. 配置 DeepSeek key
cp .env.example .env   # 然后把 DEEPSEEK_API_KEY 填进去

# 4. 只测检索（不需要 DeepSeek key）
./.venv/bin/python -m scripts.test_retrieval

# 5. 启动服务
./.venv/bin/uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000/docs 交互测试
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | /health | 健康检查 + 当前库内片段数 |
| POST | /ingest | 导入 `docs/` 下的 .md/.txt/.pdf |
| POST | /ask    | 提问，返回答案 + 引用来源 |

## 检索优化实验（已完成）

自建 12 条评测集（问题→正确文档），指标 recall@1：

| 检索方式 | recall@1 |
|----------|----------|
| 纯向量   | 92% (11/12) |
| 混合(BM25+向量, RRF 融合) | **100% (12/12)** |

典型改进案例：问「Why do I get a 422 status code?」时，纯向量被语义带偏到
`error_handling.md`，混合检索靠 BM25 精确匹配 `422` 正确命中 `path_params.md`。

运行：`./.venv/bin/python -m scripts.evaluate`

## 测试

用假 embedding + 临时向量库，不依赖 Ollama / DeepSeek：

```bash
./.venv/bin/python -m pytest -q
```

覆盖：切分逻辑、BM25 零分过滤、**重复入库幂等**、source 相对路径去冲突、混合检索冒烟。

## Roadmap（简历深度点，按顺序做）

- [x] **P0** 混合检索：BM25 + 向量（RRF 融合），提升召回
- [x] **P0** 评测集：真实问题标注正确来源，测 recall@k
- [ ] **P0** 扩充文档到 30+ 篇、评测集到 40+ 条，让数字更有说服力
- [ ] **P1** rerank：对候选片段重排，对比命中率提升
- [ ] **P1** 父子分块：小块检索、大块喂给模型
- [ ] **P1** Redis：缓存高频 query 的答案
- [ ] **P2** Agent 路由：区分「知识问答 / API 查询」走不同路径
- [ ] **P2** MySQL：存用户问答记录；Docker 一键部署

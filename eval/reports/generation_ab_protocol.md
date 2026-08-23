# 协议：hybrid@4 vs hybrid@6 最终答案盲测（dev-only）

## 目的与范围

- **只比较 k=4 与 k=6 的最终生成回答质量**，用于决定默认取多少条上下文。
  **不涉及 rerank**、不改变检索排序。
- **dev-only**：仅使用 `eval/dev.jsonl`（24 题）；**不是 test 泛化结论**，不读取 `eval/test.jsonl`。
- 每题只生成一次 hybrid top6 排名；**k4 严格取该 top6 的前 4，k6 取前 6**，
  两组唯一差别是“上下文数量”，system prompt / 问题格式 / 模型 / 温度完全相同。

## 调用预算

- 计划未来真实调用 **48 次**（24 题 × 2 组）。
- **本阶段（第一阶段）实际真实调用 = 0**：默认 dry-run / cache-only，不发起任何 DeepSeek 调用。
- 后续真实调用需**再次取得用户明确批准**，并显式 `--allow-api`。

## 生成配置

- system prompt 与上下文格式与 `app/rag.py` 一致（本阶段不修改 rag/llm）；
  `PROMPT_VERSION = genab-v1`；评测 `temperature = 0`（尽量隔离随机性）。
- 缓存键覆盖：model、temperature、prompt version、system prompt、question、**实际上下文全文**。
  因此 k4 与 k6（上下文不同）必然是不同缓存键。
- 缓存与回答只写入 `eval/outputs/generation_ab/`（已 gitignore）；**绝不触碰 `eval/rerank_cache`**。

## 文件与流程（均为 `{experiment_id, <data>}` 顶层包裹）

- `plan.json` → `{experiment_id, plan:[...]}`：每题 k4/k6 的 source_ids、上下文、cache_key。
- `ab_mapping.json` → `{experiment_id, seed, note, mapping:{...}}`：A/B→k4/k6 **真值**，
  固定种子，供解盲；**评分完成前勿看**。
- `review_packet.json` → `{experiment_id, packet:[...]}`（评判材料）：每题 `question`、
  `expected_source`、`evidence_quote`、`answer_a`、`answer_b`、`source_ids_a`/`source_ids_b`、
  `context_a`/`context_b`、`telemetry_a`/`telemetry_b`——**只标 A/B，不写 k4/k6**。
- `review_scores.json` → `{experiment_id, scores:[...]}`（**人工填写**）：每题
  `a_correctness`/`b_correctness`（0/1/2）、`a_faithfulness`/`b_faithfulness`（0/1）、
  `a_citation_correct`/`b_citation_correct`（0/1）、`reviewer_notes`。程序只生成空脚手架。
- `summary.json`（校验完整后生成，含 `experiment_id`）：见「汇总指标」。
- `run_status.json`：每次 prepare 写入 `experiment_id`、`api_attempts`、`api_successes`、
  `api_errors`、`answers_ready`、稳定 `error_type`（不含 key/完整异常文本；异常时也会写）。

## experiment_id 与错配防护

- `experiment_id = hash(dev 数据、model、temperature、prompt version、SYSTEM、k4/k6、seed、
  完整 plan cache_key 列表、A/B mapping)`；plan/mapping/packet/scores 记录同一 id。
- **summarize 前校验四者 experiment_id 完全一致**，否则非零退出、不生成 summary。
- **prepare 在检索与任何写入之前**检查 `review_scores.json`：若它不是“当前 24 题的原始空白
  脚手架”（含任意评分/备注、非法 JSON、缺字段、额外字段、缺题或重复题），且**无 `--force`** →
  **立即非零退出、不检索、不覆盖 plan/mapping/packet/scores**，避免评分与新产物错配。
- `--force` 会**废弃整次评测状态并一起重建全部产物**（含已填评分），并给出明确提示。
- CLI：`--summarize` 与 `--allow-api`/`--force` **不可同用**，冲突时明确报错。

**A/B 位置平衡**：24 题严格 **12 题 A=k4、12 题 A=k6**——先造 12 个 k4 + 12 个 k6 标签，
再用固定种子 shuffle（非逐题独立抛硬币）。

**盲测诚实说明**：这是**隐藏明确 k 标签**的标签盲测。由于 k4/k6 的上下文**片段数量不同**，
评审者可能据此**推断**配置，**因此不宣称完全双盲**；A/B→k 真值映射单独保存、评分前不看。

- dry-run 无答案时，`answer_*` 为 `null`——**只输出结构，不伪造回答或指标**。

## 评分标准

- **correctness（正确性，0/1/2）**：`0` 错误；`1` 部分正确（要点不全或有次要错误）；
  `2` 完全正确（准确且完整回答了问题）。
- **faithfulness（忠实性，0/1）**：回答是否**只依据所给上下文**、无超出上下文的臆造。
  `1` 完全有据；`0` 含上下文中不存在的断言（幻觉）。
- **citation correctness（引用正确性，0/1）**：末尾 `[来源: 文件名#片段号]` 标注是否
  **确实对应支持答案的片段**。`1` 引用正确且相关；`0` 缺失/错误/张冠李戴。

## 评分严格校验（汇总前）

- 必须**正好 24 个唯一 dev id**；每题 A/B 答案都存在（共 **48 个有效答案**）。
- `correctness` 只能是整数 **0/1/2**（**bool 不算整数评分**）；
  `faithfulness`、`citation_correct` 只能是整数 **0/1**（拒绝字符串、3、负数、缺字段）。
- 任一不满足 → **非零退出，不生成 summary**。

## 汇总指标（`summary.json`）

- 单条 `overall_pass` = `correctness==2 且 faithfulness==1 且 citation_correct==1`。
- 解盲后按 **k4/k6 分别报告**：overall pass 数/率、平均 correctness、faithfulness 通过率、
  citation 通过率；以及成本/延迟：平均输入 token、平均输出 token、总 token、平均延迟、p50、p95。
- **逐题配对比较**：k6 win / k4 win / tie（按 overall_pass），并列出对应 dev id；
  `net_k6_wins = |k6_win| - |k4_win|`。

## 决策门槛（提前冻结，避免看数调参）

- 进入「**可能作为默认**」候选需同时满足：
  * `net_k6_wins >= 2`；
  * **0 个 overall-pass regression**（不存在 k4 pass 而 k6 fail 的题）；
  * k6 的 **faithfulness 失败数、citation 失败数均不增加**（相对 k4）。
- 定性方向：k6 无明显改善 → 倾向 **k4**；k6 明显减少错误且成本/延迟可接受 → 倾向 **k6**。
- **成本与延迟上限须在真实调用前由用户明确冻结**。
- 工具**只报告是否满足门槛**（`k6_candidate` 布尔），**不自动宣称上线**。
- 判定以人工评分解盲汇总为准；本协议在生成/评分前冻结，不得事后放宽。

## 安全边界（本阶段）

- 真实 DeepSeek 调用 = 0；不使用 `--allow-api`；不读取/显示/输出 `.env` 中的 key。
- 不读取或运行 `eval/test.jsonl`；不运行现有实验脚本。
- 不 reset/rebuild/修改真实 Chroma 索引；不删除或修改 `eval/rerank_cache`。
- 不修改 `/ask`、`app/rag.py`、`app/llm.py` 或其它生产行为。
- 后续真实调用（48 次）必须**再次获得用户明确批准**。

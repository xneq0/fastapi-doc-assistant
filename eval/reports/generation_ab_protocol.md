# 协议：hybrid@4 vs hybrid@6 最终答案分阶段盲评（dev-only）

## 目的与范围

- **只比较 k=4 与 k=6 的最终生成回答质量**，用于决定默认取多少条上下文。
  **不涉及 rerank**、不改变检索排序。
- **dev-only**：仅使用 `eval/dev.jsonl`（24 题）；**不是 test 泛化结论**，不读取 `eval/test.jsonl`。
- 每题只生成一次 hybrid top6 排名；**k4 严格取该 top6 的前 4，k6 取前 6**，
  两组唯一差别是“上下文数量”，system prompt / 问题格式 / 模型 / thinking / 输出上限 /
  温度完全相同。

## 调用记录与预算

- 2026-08-26 已经用户明确批准并完成真实调用 **48 次**（24 题 × 2 组），48/48 成功。
- 后续重建评审材料只读现有严格缓存，**不重新生成答案、不发起 API 调用**。
- 任何新增真实调用仍需再次取得用户明确批准，并显式 `--allow-api`。

## 生成配置

- system prompt 与上下文格式与 `app/rag.py` 一致（本阶段不修改 rag/llm）；生成配置在
  dev A/B 工具内独立冻结为 `deepseek-v4-flash`、`thinking=disabled`、`max_tokens=1024`、
  `temperature=0`、`PROMPT_VERSION=genab-v2`。这延续旧 `deepseek-chat` 的非思考实验语义，
  同时避免 V4 默认 thinking 改变成本、延迟和回答行为；结论只对该冻结配置成立。
- 缓存键覆盖：model、thinking、max_tokens、temperature、prompt version、system prompt、
  question、**实际上下文全文**。
  因此 k4 与 k6（上下文不同）必然是不同缓存键。
- 缓存与回答只写入 `eval/outputs/generation_ab/`（已 gitignore）；**绝不触碰 `eval/rerank_cache`**。

## 文件与分阶段流程

- `plan.json` → `{experiment_id, plan:[...]}`：每题 k4/k6 的 source_ids、上下文、cache_key。
- `ab_mapping.json` → `{experiment_id, seed, note, mapping:{...}}`：A/B→k4/k6 **真值**，
  固定种子，供解盲；**评分完成前勿看**。
- 第一阶段 `review_packet.json`：每题只显示 `question`、gold evidence、A/B 答案，以及
  `citation_evidence_a/b`。工具只从答案中显式写出的 citation 提取 ID，再到两侧共享的 top6
  参考集合解析对应 chunk 正文；不展示完整 reference context/source_ids、检索排名或总数。
  citation 缺失标记为 `missing`；格式错误或无法解析标记为 `unresolved`，不得静默丢弃。
  主评审包禁止出现侧别 context、侧别 source_ids 或 telemetry，因此不能从片段数或输入 token
  直接识别 k4/k6。
- 第一阶段 `review_scores.json`：只填写 A/B 的 correctness（0/1/2）、citation correctness
  （0/1）和备注；不含 faithfulness 字段。
- 第一阶段 24 题完整后运行 `--prepare-faithfulness`：校验并哈希锁定主评分，才生成
  `faithfulness_packet.json` 与 `faithfulness_scores.json`。审计包显示两侧真实生成上下文，
  只评分 faithfulness（0/1）；不包含 telemetry。
- `--summarize` 前须校验主评分 hash 未变化、两个阶段 experiment_id 一致、两阶段答案一致，
  并从严格缓存内部重建 telemetry；telemetry 不进入人工评审材料。
- `summary.json`（校验完整后生成，含 `experiment_id`）：见「汇总指标」。
- `run_status.json`：每次 prepare 写入 `experiment_id`、`api_attempts`、`api_successes`、
  `api_errors`、`answers_ready`、稳定 `error_type` 和非敏感 `generation_config`
  （不含 key/完整异常文本；异常时也会写）。

## experiment_id 与错配防护

- `experiment_id = hash(dev 数据、model、thinking、max_tokens、temperature、prompt version、
  SYSTEM、k4/k6、seed、
  完整 plan cache_key 列表、A/B mapping)`；plan/mapping/packet/scores 记录同一 id。
- **summarize 前校验六份两阶段产物 experiment_id 完全一致**，否则非零退出、不生成 summary。
- **prepare 在检索与任何写入之前**检查 `review_scores.json`：若它不是“当前 24 题的原始空白
  脚手架”（含任意评分/备注、非法 JSON、缺字段、额外字段、缺题或重复题），且**无 `--force`** →
  **立即非零退出、不检索、不覆盖 plan/mapping/packet/scores**，避免评分与新产物错配。
- `--force` 会**废弃整次评测状态并一起重建全部产物**（含已填评分），并给出明确提示。
- CLI：`--summarize`、`--prepare-faithfulness` 互斥，且与 `--allow-api`/`--force`
  **不可同用**，冲突时明确报错。

**A/B 位置平衡**：24 题严格 **12 题 A=k4、12 题 A=k6**——先造 12 个 k4 + 12 个 k6 标签，
再用固定种子 shuffle（非逐题独立抛硬币）。

**盲评诚实说明**：第一阶段隐藏明确 k 标签、完整参考上下文、侧别上下文/source_ids 和 telemetry，
但回答本身的长度、引用或内容特征仍可能让评审者猜测条件，因此不宣称完全双盲。
第二阶段为了精确判断“是否只依据真实生成上下文”，必须显示两侧上下文，片段数会暴露 k；
此时 correctness/citation 主评分已通过 hash 锁定，不能事后修改后再汇总。

- dry-run 无答案时，`answer_*` 为 `null`——**只输出结构，不伪造回答或指标**。

## 评分标准

- **correctness（正确性，0/1/2）**：`0` 错误；`1` 部分正确（要点不全或有次要错误）；
  `2` 完全正确（准确且完整回答了问题）。
- **faithfulness（忠实性，0/1）**：回答是否**只依据所给上下文**、无超出上下文的臆造。
  `1` 完全有据；`0` 含上下文中不存在的断言（幻觉）。
- **citation correctness（引用正确性，0/1）**：末尾 `[来源: 文件名#片段号]` 标注是否
  **确实对应支持答案的片段**。`1` 引用正确且相关；`0` 缺失/错误/张冠李戴。

## 评分严格校验（汇总前）

- 两阶段均须**正好 24 个唯一 dev id**；每题 A/B 答案都存在（共 **48 个有效答案**）。
- 第一阶段 `correctness` 只能是整数 **0/1/2**，`citation_correct` 只能是整数 **0/1**；
  第二阶段 `faithfulness` 只能是整数 **0/1**（bool 不算整数；拒绝字符串、越界值、缺字段）。
- faithfulness 阶段记录的主评分 hash 必须与当前 `review_scores.json` 完全一致。
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

- 本次评审材料重建真实 DeepSeek 调用 = 0；不使用 `--allow-api`；不读取/显示/输出 `.env` 中的 key。
- 不读取或运行 `eval/test.jsonl`；不运行现有实验脚本。
- 不 reset/rebuild/修改真实 Chroma 索引；不删除或修改 `eval/rerank_cache`。
- 不修改 `/ask`、`app/rag.py`、`app/llm.py` 或其它生产行为。
- 任何后续真实调用必须**再次获得用户明确批准**。

"""dev-only 的 hybrid@4 vs hybrid@6 最终答案盲测工具（含数据一致性加固）。

目的：比较同一问题在 k=4 与 k=6 上下文下的**最终生成回答**质量，不涉及 rerank。
- 只读 eval/dev.jsonl（24 题）；不提供 split 参数，不引用冻结的 test 集。
- 每题只生成一次 hybrid top6；k4 严格取该 top6 的前 4，k6 取前 6，两组唯一差别是上下文数量。
- 产物均带同一 experiment_id（防止评分与新 plan/mapping/packet 错配）。
- 三种阶段模式：
  * prepare（默认，dry-run/cache-only）：生成 plan / ab_mapping / review_packet /
    review_scores 主评脚手架；主评包只含 gold evidence、答案及其 citation 对应 chunk，
    不展示完整 reference、侧别上下文或 telemetry；
    真实 DeepSeek 调用仅在显式 --allow-api 时发生。
  * --prepare-faithfulness：主评分完整后哈希锁定，才生成真实侧别上下文审计包与评分脚手架；
    不检索、不建 client、不调 API。
  * --summarize：只读已填写的评分与 A/B mapping，解盲后汇总；不检索、不建 client、不调 API。
- 缓存与产物只写入 eval/outputs/（已 gitignore）；绝不触碰 eval/rerank_cache。

安全：无 key（生成时）/缓存或评分不完整或非法/experiment_id 不一致/API 异常 → 非零退出。
OpenAI client 惰性创建；空 key 也可导入与 dry-run。
"""
import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

from app import config

# ── 生成配置（eval-only 冻结；本阶段不修改生产 rag/llm）─────────────
PROMPT_VERSION = "genab-v2"
EVAL_MODEL = "deepseek-v4-flash"
EVAL_THINKING = "disabled"
EVAL_MAX_TOKENS = 1024
EVAL_TEMPERATURE = 0.0
SYSTEM = (
    "你是一个开发者文档问答助手。只依据给定的【文档片段】回答问题，"
    "不要编造。若文档中找不到答案，明确说『文档中未提及』。"
    "回答末尾用 [来源: 文件名#片段号] 的形式标注你引用了哪些片段。"
)

K_SMALL, K_BIG = 4, 6
EXPECTED_ANSWERS = 48
AB_SEED = 20260823
K_LABEL = {"k4": K_SMALL, "k6": K_BIG}
PRIMARY_SCORE_FIELDS = ("a_correctness", "b_correctness",
                        "a_citation_correct", "b_citation_correct")
FAITH_SCORE_FIELDS = ("a_faithfulness", "b_faithfulness")
PRIMARY_SCAFFOLD_KEYS = {"id", *PRIMARY_SCORE_FIELDS, "reviewer_notes"}
FAITH_SCAFFOLD_KEYS = {"id", *FAITH_SCORE_FIELDS, "reviewer_notes"}

DEV_PATH = Path("eval/dev.jsonl")
OUTPUT_DIR = Path("eval/outputs/generation_ab")
PLAN_FILE, MAPPING_FILE = "plan.json", "ab_mapping.json"
PACKET_FILE, SCORES_FILE = "review_packet.json", "review_scores.json"
FAITH_PACKET_FILE, FAITH_SCORES_FILE = "faithfulness_packet.json", "faithfulness_scores.json"
SUMMARY_FILE, STATUS_FILE = "summary.json", "run_status.json"


# ── prompt / cache key ─────────────────────────────────────────────
def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['source']}#{h['chunk_index']}]\n{h['text']}" for h in hits)


def build_user(context: str, question: str) -> str:
    return f"【文档片段】\n{context}\n\n【问题】\n{question}"


def cache_key(question: str, context: str, model: str = EVAL_MODEL,
              thinking: str = EVAL_THINKING, max_tokens: int = EVAL_MAX_TOKENS,
              temperature: float = EVAL_TEMPERATURE, system: str = SYSTEM,
              prompt_version: str = PROMPT_VERSION) -> str:
    payload = "\n\x1e\n".join([model, thinking, str(max_tokens), repr(float(temperature)),
                               prompt_version, system, question, context])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── 计划 / 前缀校验 ────────────────────────────────────────────────
def build_plan(dev_items: list[dict], retrieve, k_small: int = K_SMALL, k_big: int = K_BIG) -> list[dict]:
    if not (0 < k_small < k_big):
        raise ValueError(f"invalid k: k_small={k_small}, k_big={k_big}（需 0<k_small<k_big）")
    plan = []
    for it in dev_items:
        top = retrieve(it["question"])
        if len(top) < k_big:
            raise ValueError(f"{it['id']}: 检索到 {len(top)} 条 < k_big={k_big}，无法构造 k6")
        top6 = top[:k_big]
        for k in (k_small, k_big):
            hits = top6[:k]
            ctx = build_context(hits)
            plan.append({
                "id": it["id"], "question": it["question"], "k": k,
                "source_ids": [f"{h['source']}#{h['chunk_index']}" for h in hits],
                "chunks": [{"source_id": f"{h['source']}#{h['chunk_index']}",
                            "text": h["text"]} for h in hits],
                "context": ctx, "context_chars": len(ctx),
                "cache_key": cache_key(it["question"], ctx),
            })
    return plan


def assert_prefix(plan: list[dict], k_small: int = K_SMALL, k_big: int = K_BIG):
    by = {}
    for p in plan:
        by.setdefault(p["id"], {})[p["k"]] = p["source_ids"]
    for _id, ks in by.items():
        small, big = ks.get(k_small), ks.get(k_big)
        if small is None or big is None or big[:k_small] != small:
            raise ValueError(f"{_id}: k{k_small} 不是 k{k_big} 的严格前缀")


# ── A/B 位置平衡映射（固定种子；12 k4 + 12 k6 再 shuffle）───────────
def make_ab_mapping(ids, seed: int = AB_SEED) -> dict:
    ids_sorted = sorted(ids)
    half = len(ids_sorted) // 2
    labels = ["k4"] * half + ["k6"] * (len(ids_sorted) - half)
    random.Random(seed).shuffle(labels)
    return {_id: {"A": a, "B": "k6" if a == "k4" else "k4"}
            for _id, a in zip(ids_sorted, labels)}


# ── experiment_id：绑定整次评测的全部关键输入 ─────────────────────
def compute_experiment_id(dev: list[dict], plan: list[dict], mapping: dict, seed: int) -> str:
    payload = json.dumps({
        "dev": dev,
        "model": EVAL_MODEL,
        "thinking": EVAL_THINKING,
        "max_tokens": EVAL_MAX_TOKENS,
        "temperature": EVAL_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "system": SYSTEM,
        "k_small": K_SMALL, "k_big": K_BIG,
        "seed": seed,
        "plan_cache_keys": [p["cache_key"] for p in plan],
        "mapping": mapping,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── telemetry / 缓存加固 ───────────────────────────────────────────
def _nonneg_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _valid_latency(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def _valid_telemetry(t) -> bool:
    return (isinstance(t, dict) and _nonneg_int(t.get("in_tokens"))
            and _nonneg_int(t.get("out_tokens")) and _valid_latency(t.get("latency")))


def _cache_dir(out_dir: Path) -> Path:
    return out_dir / "cache"


def load_cache_record(out_dir: Path, key: str, expected_id=None, expected_k=None):
    """按生成配置、题目身份、答案与 telemetry 严格重校验缓存。"""
    p = _cache_dir(out_dir) / f"{key}.json"
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("cache_key") != key:
        return None
    if rec.get("model") != EVAL_MODEL:
        return None
    if rec.get("thinking") != EVAL_THINKING:
        return None
    if rec.get("max_tokens") != EVAL_MAX_TOKENS:
        return None
    if rec.get("prompt_version") != PROMPT_VERSION:
        return None
    if rec.get("temperature") != EVAL_TEMPERATURE:
        return None
    if expected_id is not None and rec.get("id") != expected_id:
        return None
    if expected_k is not None and rec.get("k") != expected_k:
        return None
    ans = rec.get("answer")
    if not (isinstance(ans, str) and ans.strip()):
        return None
    if not _valid_telemetry(rec.get("telemetry")):
        return None
    return rec


def load_answer(out_dir: Path, key: str, expected_id=None, expected_k=None):
    rec = load_cache_record(out_dir, key, expected_id, expected_k)
    return rec["answer"] if rec else None


# ── 生成（惰性 client；返回 answer + telemetry）─────────────────────
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


def generate_answer(system: str, user: str, chat_fn=None):
    if chat_fn is not None:
        return chat_fn(system, user)
    t0 = time.time()
    resp = _get_client().chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=EVAL_TEMPERATURE,
        max_tokens=EVAL_MAX_TOKENS,
        extra_body={"thinking": {"type": EVAL_THINKING}},
    )
    dt = time.time() - t0
    u = resp.usage
    return resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens, round(dt, 3)


def _key_ok() -> bool:
    k = config.DEEPSEEK_API_KEY
    return bool(k) and k.strip() != "sk-你的key" and k.isascii()


# ── 分阶段 review packet / scores 脚手架 ──────────────────────────
_CITATION_BLOCK_RE = re.compile(r"\[来源\s*[:：]\s*([^\]]*)\]")
_CITATION_ID_RE = re.compile(r"([^\s,，;；\[\]]+#\d+)")


def build_citation_evidence(answer: str | None, reference_chunks: dict[str, str]) -> dict:
    """只解析答案显式 citation；不使用侧别生成上下文或 k 标签。"""
    blocks = _CITATION_BLOCK_RE.findall(answer or "")
    if not blocks:
        return {"status": "missing", "items": []}

    items, seen = [], set()
    invalid_blocks = []
    for raw in blocks:
        ids = [m.strip("`'\"()<>。.！!？?") for m in _CITATION_ID_RE.findall(raw)]
        if not ids:
            invalid_blocks.append(raw.strip())
            continue
        for citation in ids:
            if citation in seen:
                continue
            seen.add(citation)
            text = reference_chunks.get(citation)
            items.append({"citation": citation,
                          "status": "resolved" if text is not None else "not_found",
                          "text": text})
    for raw in invalid_blocks:
        items.append({"citation": raw, "status": "invalid_format", "text": None})
    resolved = bool(items) and all(it["status"] == "resolved" for it in items)
    return {"status": "resolved" if resolved else "unresolved", "items": items}


def build_packet(plan: list[dict], mapping: dict, dev_by_id: dict, out_dir: Path):
    """第一阶段主评审包：只给 gold evidence、答案和答案实际引用片段。"""
    by = {(p["id"], p["k"]): p for p in plan}
    ids = sorted({p["id"] for p in plan})
    packet, n_answers = [], 0
    for _id in ids:
        m = mapping[_id]
        pa, pb = by[(_id, K_LABEL[m["A"]])], by[(_id, K_LABEL[m["B"]])]
        ra = load_cache_record(out_dir, pa["cache_key"], _id, pa["k"])
        rb = load_cache_record(out_dir, pb["cache_key"], _id, pb["k"])
        ans_a = ra["answer"] if ra else None
        ans_b = rb["answer"] if rb else None
        n_answers += (ans_a is not None) + (ans_b is not None)
        dev = dev_by_id[_id]
        reference = by[(_id, K_BIG)]
        reference_chunks = {c["source_id"]: c["text"] for c in reference["chunks"]}
        packet.append({
            "id": _id, "question": pa["question"],
            "expected_source": dev.get("expected_source"),
            "evidence_quote": dev.get("evidence_quote"),
            "answer_a": ans_a, "answer_b": ans_b,
            "citation_evidence_a": build_citation_evidence(ans_a, reference_chunks),
            "citation_evidence_b": build_citation_evidence(ans_b, reference_chunks),
        })
    return packet, n_answers


def build_faithfulness_packet(plan: list[dict], mapping: dict, out_dir: Path):
    """第二阶段忠实性审计包：主评分锁定后才生成；暴露实际两侧生成上下文。"""
    by = {(p["id"], p["k"]): p for p in plan}
    packet = []
    for _id in sorted({p["id"] for p in plan}):
        m = mapping[_id]
        pa, pb = by[(_id, K_LABEL[m["A"]])], by[(_id, K_LABEL[m["B"]])]
        ra = load_cache_record(out_dir, pa["cache_key"], _id, pa["k"])
        rb = load_cache_record(out_dir, pb["cache_key"], _id, pb["k"])
        packet.append({
            "id": _id, "question": pa["question"],
            "answer_a": ra["answer"] if ra else None,
            "answer_b": rb["answer"] if rb else None,
            "source_ids_a": pa["source_ids"], "source_ids_b": pb["source_ids"],
            "context_a": pa["context"], "context_b": pb["context"],
        })
    return packet


def build_telemetry_packet(plan: list[dict], mapping: dict, out_dir: Path):
    """汇总时从严格校验过的缓存重建 telemetry；不写入评审材料。"""
    by = {(p["id"], p["k"]): p for p in plan}
    packet = []
    for _id in sorted({p["id"] for p in plan}):
        m = mapping[_id]
        pa, pb = by[(_id, K_LABEL[m["A"]])], by[(_id, K_LABEL[m["B"]])]
        ra = load_cache_record(out_dir, pa["cache_key"], _id, pa["k"])
        rb = load_cache_record(out_dir, pb["cache_key"], _id, pb["k"])
        packet.append({
            "id": _id,
            "telemetry_a": ra["telemetry"] if ra else None,
            "telemetry_b": rb["telemetry"] if rb else None,
        })
    return packet


def scaffold_scores(ids, fields=PRIMARY_SCORE_FIELDS) -> list[dict]:
    out = []
    for _id in sorted(ids):
        row = {"id": _id}
        for f in fields:
            row[f] = None
        row["reviewer_notes"] = ""
        out.append(row)
    return out


def is_pristine_scaffold(path: Path, expected_ids, fields=PRIMARY_SCORE_FIELDS,
                         scaffold_keys=PRIMARY_SCAFFOLD_KEYS) -> bool:
    """判断 review_scores.json 是否为“当前 24 题的原始空白脚手架”。任何偏离都返回 False。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("scores"), list):
        return False
    rows = data["scores"]
    ids = [r.get("id") for r in rows if isinstance(r, dict)]
    if len(rows) != len(expected_ids) or len(ids) != len(rows):
        return False
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        return False
    for r in rows:
        if set(r.keys()) != scaffold_keys:                 # 缺字段或额外字段
            return False
        if any(r[f] is not None for f in fields):          # 已有评分
            return False
        if r.get("reviewer_notes", "") != "":              # 有备注
            return False
    return True


# ── 校验：mapping / packet / scores ────────────────────────────────
def validate_mapping(mapping, expected_ids) -> list[str]:
    errors = []
    if not isinstance(mapping, dict):
        return ["mapping 不是 dict"]
    if set(mapping) != set(expected_ids):
        missing = set(expected_ids) - set(mapping)
        extra = set(mapping) - set(expected_ids)
        if missing:
            errors.append(f"mapping 缺题: {sorted(missing)}")
        if extra:
            errors.append(f"mapping 多余题: {sorted(extra)}")
    a_k4 = a_k6 = 0
    for _id, v in mapping.items():
        if not isinstance(v, dict) or {v.get("A"), v.get("B")} != {"k4", "k6"}:
            errors.append(f"{_id}: A/B 标签非法（须恰好覆盖 k4/k6）: {v!r}")
            continue
        a_k4 += v["A"] == "k4"; a_k6 += v["A"] == "k6"
    if not errors and (a_k4 != len(expected_ids) // 2 or a_k6 != len(expected_ids) - len(expected_ids) // 2):
        errors.append(f"A 位置未严格平衡：A=k4 {a_k4}，A=k6 {a_k6}")
    return errors


def _nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _valid_citation_evidence(value) -> bool:
    if not isinstance(value, dict) or set(value) != {"status", "items"}:
        return False
    status, items = value["status"], value["items"]
    if status not in {"missing", "resolved", "unresolved"} or not isinstance(items, list):
        return False
    if status == "missing":
        return items == []
    if not items:
        return False
    for item in items:
        if not isinstance(item, dict) or set(item) != {"citation", "status", "text"}:
            return False
        if not isinstance(item["citation"], str) or item["status"] not in {
                "resolved", "not_found", "invalid_format"}:
            return False
        if item["status"] == "resolved":
            if not isinstance(item["text"], str):
                return False
        elif item["text"] is not None:
            return False
    all_resolved = all(item["status"] == "resolved" for item in items)
    return (status == "resolved") == all_resolved


def validate_packet(packet, dev) -> list[str]:
    """校验第一阶段盲评包；禁止出现会直接泄露条件的侧别字段。"""
    errors = []
    if not isinstance(packet, list):
        return ["packet 不是列表"]
    expected_ids = [it["id"] for it in dev]
    ids = [p.get("id") for p in packet if isinstance(p, dict)]
    if len(ids) != len(set(ids)):
        errors.append("packet 存在重复 id")
    if set(ids) != set(expected_ids):
        errors.append("packet id 与 dev 不一致")
    if len(packet) != len(expected_ids):
        errors.append(f"packet 条数 {len(packet)} != {len(expected_ids)}")
    for p in packet:
        if not isinstance(p, dict):
            errors.append(f"非法 packet 行: {p!r}"); continue
        pid = p.get("id")
        forbidden = {"source_ids_a", "source_ids_b", "context_a", "context_b",
                     "telemetry_a", "telemetry_b", "reference_source_ids",
                     "reference_context"} & set(p)
        if forbidden:
            errors.append(f"{pid}: 主评审包泄露侧别条件字段: {sorted(forbidden)}")
        for f in ("question", "expected_source", "evidence_quote"):
            if not isinstance(p.get(f), str):
                errors.append(f"{pid}.{f} 必须是字符串")
        for f in ("answer_a", "answer_b"):
            if not _nonempty_str(p.get(f)):
                errors.append(f"{pid}.{f} 必须是 strip 后非空字符串")
        for f in ("citation_evidence_a", "citation_evidence_b"):
            if not _valid_citation_evidence(p.get(f)):
                errors.append(f"{pid}.{f} 结构/状态非法")
    return errors


def validate_faithfulness_packet(packet, dev) -> list[str]:
    errors = []
    if not isinstance(packet, list):
        return ["faithfulness packet 不是列表"]
    expected_ids = [it["id"] for it in dev]
    ids = [p.get("id") for p in packet if isinstance(p, dict)]
    if len(ids) != len(set(ids)):
        errors.append("faithfulness packet 存在重复 id")
    if set(ids) != set(expected_ids) or len(packet) != len(expected_ids):
        errors.append("faithfulness packet id/条数与 dev 不一致")
    for p in packet:
        if not isinstance(p, dict):
            errors.append(f"非法 faithfulness packet 行: {p!r}"); continue
        pid = p.get("id")
        for f in ("question", "context_a", "context_b"):
            if not isinstance(p.get(f), str):
                errors.append(f"{pid}.{f} 必须是字符串")
        for f in ("answer_a", "answer_b"):
            if not _nonempty_str(p.get(f)):
                errors.append(f"{pid}.{f} 必须是 strip 后非空字符串")
        for f in ("source_ids_a", "source_ids_b"):
            v = p.get(f)
            if not (isinstance(v, list) and v and all(isinstance(x, str) and x for x in v)):
                errors.append(f"{pid}.{f} 必须是非空字符串列表")
    return errors


def validate_telemetry_packet(packet, expected_ids) -> list[str]:
    if not isinstance(packet, list):
        return ["telemetry packet 不是列表"]
    errors = []
    ids = [p.get("id") for p in packet if isinstance(p, dict)]
    if len(packet) != len(expected_ids) or len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        errors.append("telemetry packet id/条数与 dev 不一致")
    for p in packet:
        if not isinstance(p, dict):
            errors.append(f"非法 telemetry packet 行: {p!r}"); continue
        for f in ("telemetry_a", "telemetry_b"):
            if not _valid_telemetry(p.get(f)):
                errors.append(f"{p.get('id')}.{f} telemetry 不完整/非法")
    return errors


def validate_plan(plan, dev, mapping) -> list[str]:
    """深度校验 plan：每题恰好 k4/k6 两条；question 与 dev 一致；k4 是 k6 前缀；
    context_chars 与实际一致；cache_key 用当前配置重算后完全一致。"""
    errors = []
    if not isinstance(plan, list):
        return ["plan 不是列表"]
    dev_by_id = {it["id"]: it for it in dev}
    by_id = {}
    for p in plan:
        if not isinstance(p, dict):
            errors.append(f"非法 plan 行: {p!r}"); continue
        by_id.setdefault(p.get("id"), {})[p.get("k")] = p
    if set(by_id) != set(dev_by_id):
        errors.append("plan id 与 dev 不一致")
    for _id, dev_it in dev_by_id.items():
        ks = by_id.get(_id, {})
        if set(ks) != {K_SMALL, K_BIG}:
            errors.append(f"{_id}: plan 未恰好包含 k{K_SMALL}/k{K_BIG} 两条"); continue
        for k, p in ks.items():
            if p.get("question") != dev_it.get("question"):
                errors.append(f"{_id} k{k}: question 与 dev 不一致")
            chunks = p.get("chunks")
            if not (isinstance(chunks, list) and len(chunks) == k
                    and all(isinstance(c, dict) and _nonempty_str(c.get("source_id"))
                            and isinstance(c.get("text"), str) for c in chunks)):
                errors.append(f"{_id} k{k}: chunks 结构/数量非法")
            elif [c["source_id"] for c in chunks] != p.get("source_ids"):
                errors.append(f"{_id} k{k}: chunks 与 source_ids 不一致")
            ctx = p.get("context")
            if not isinstance(ctx, str) or p.get("context_chars") != len(ctx):
                errors.append(f"{_id} k{k}: context_chars 与实际长度不符")
            else:
                if isinstance(chunks, list) and all(isinstance(c, dict) for c in chunks):
                    rebuilt = "\n\n".join(f"[{c.get('source_id')}]\n{c.get('text')}" for c in chunks)
                    if ctx != rebuilt:
                        errors.append(f"{_id} k{k}: context 与 chunks 重建结果不一致")
                if p.get("cache_key") != cache_key(dev_it.get("question", ""), ctx):
                    errors.append(f"{_id} k{k}: cache_key 重算不一致")
        s4, s6 = ks[K_SMALL].get("source_ids"), ks[K_BIG].get("source_ids")
        if not (isinstance(s4, list) and isinstance(s6, list) and s6[:K_SMALL] == s4):
            errors.append(f"{_id}: k{K_SMALL} source_ids 不是 k{K_BIG} 的前缀")
    return errors


def validate_packet_derived(packet, dev, plan, mapping) -> list[str]:
    """校验主评审包的 gold 字段与 citation evidence 均可由源数据重建。"""
    errors = []
    dev_by_id = {it["id"]: it for it in dev}
    plan_by = {(p["id"], p["k"]): p for p in plan if isinstance(p, dict)}
    for pk in packet:
        if not isinstance(pk, dict):
            continue
        _id = pk.get("id")
        dev_it = dev_by_id.get(_id)
        reference = plan_by.get((_id, K_BIG))
        if dev_it is None or reference is None:
            errors.append(f"{_id}: packet 无法对应 dev/plan"); continue
        if pk.get("question") != dev_it.get("question"):
            errors.append(f"{_id}: packet.question 与 dev 不一致")
        if pk.get("expected_source") != dev_it.get("expected_source"):
            errors.append(f"{_id}: packet.expected_source 与 dev 不一致")
        if pk.get("evidence_quote") != dev_it.get("evidence_quote"):
            errors.append(f"{_id}: packet.evidence_quote 与 dev 不一致")
        reference_chunks = {c["source_id"]: c["text"] for c in reference.get("chunks", [])}
        for side in ("a", "b"):
            expected = build_citation_evidence(pk.get(f"answer_{side}"), reference_chunks)
            if pk.get(f"citation_evidence_{side}") != expected:
                errors.append(f"{_id}: packet.citation_evidence_{side} 无法由答案/top{K_BIG} 重建")
    return errors


def validate_faithfulness_packet_derived(packet, dev, plan, mapping) -> list[str]:
    """校验忠实性审计包的两侧真实生成上下文。"""
    errors = []
    dev_by_id = {it["id"]: it for it in dev}
    plan_by = {(p["id"], p["k"]): p for p in plan if isinstance(p, dict)}
    for pk in packet:
        if not isinstance(pk, dict):
            continue
        _id = pk.get("id")
        m = mapping.get(_id)
        dev_it = dev_by_id.get(_id)
        if m is None or dev_it is None or {m.get("A"), m.get("B")} != {"k4", "k6"}:
            errors.append(f"{_id}: faithfulness packet 无法对应 dev/mapping"); continue
        pa = plan_by.get((_id, K_LABEL[m["A"]]))
        pb = plan_by.get((_id, K_LABEL[m["B"]]))
        if pa is None or pb is None:
            errors.append(f"{_id}: faithfulness packet 对应 plan 项缺失"); continue
        if pk.get("question") != dev_it.get("question"):
            errors.append(f"{_id}: faithfulness packet.question 与 dev 不一致")
        for side, p in (("a", pa), ("b", pb)):
            if pk.get(f"source_ids_{side}") != p.get("source_ids"):
                errors.append(f"{_id}: faithfulness packet.source_ids_{side} 与 plan 不一致")
            if pk.get(f"context_{side}") != p.get("context"):
                errors.append(f"{_id}: faithfulness packet.context_{side} 与 plan 不一致")
    return errors


def validate_packets_agree(primary_packet, faith_packet) -> list[str]:
    """两阶段问题与答案必须完全一致，防止锁定后替换答案。"""
    errors = []
    primary_by = {p.get("id"): p for p in primary_packet if isinstance(p, dict)}
    faith_by = {p.get("id"): p for p in faith_packet if isinstance(p, dict)}
    if set(primary_by) != set(faith_by):
        return ["主评审包与 faithfulness packet id 不一致"]
    for _id, p in primary_by.items():
        f = faith_by[_id]
        for field in ("question", "answer_a", "answer_b"):
            if p.get(field) != f.get(field):
                errors.append(f"{_id}: 两阶段 {field} 不一致")
    return errors


def _valid_generation(ans, in_tok, out_tok, lat) -> bool:
    """真实生成返回值写缓存前的严格校验。"""
    return (isinstance(ans, str) and ans.strip() != ""
            and _nonneg_int(in_tok) and _nonneg_int(out_tok) and _valid_latency(lat))


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_score_rows(scores, expected_ids, fields, allowed, label) -> list[str]:
    errors = []
    if not isinstance(scores, list):
        return [f"{label} scores 不是列表"]
    ids = [s.get("id") for s in scores if isinstance(s, dict)]
    if len(ids) != len(set(ids)):
        errors.append(f"{label} scores 存在重复 id")
    if set(ids) != set(expected_ids):
        missing = set(expected_ids) - set(ids)
        extra = set(ids) - set(expected_ids)
        if missing:
            errors.append(f"{label} scores 缺题: {sorted(missing)}")
        if extra:
            errors.append(f"{label} scores 多余 id: {sorted(extra)}")
    if len(scores) != len(expected_ids):
        errors.append(f"{label} 评分条数 {len(scores)} != {len(expected_ids)}")
    for s in scores:
        if not isinstance(s, dict):
            errors.append(f"非法评分行: {s!r}"); continue
        sid = s.get("id")
        for f in fields:
            v = s.get(f, "__MISSING__")
            valid_values = allowed[f]
            if not _is_int(v) or v not in valid_values:
                errors.append(f"{sid}.{f} 非法（须整数 {sorted(valid_values)}）: {v!r}")
    return errors


def validate_scores(scores, expected_ids) -> list[str]:
    allowed = {"a_correctness": {0, 1, 2}, "b_correctness": {0, 1, 2},
               "a_citation_correct": {0, 1}, "b_citation_correct": {0, 1}}
    return _validate_score_rows(scores, expected_ids, PRIMARY_SCORE_FIELDS, allowed, "primary")


def validate_faithfulness_scores(scores, expected_ids) -> list[str]:
    allowed = {"a_faithfulness": {0, 1}, "b_faithfulness": {0, 1}}
    return _validate_score_rows(scores, expected_ids, FAITH_SCORE_FIELDS, allowed, "faithfulness")


def primary_scores_hash(scores) -> str:
    payload = json.dumps(scores, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def combine_scores(primary_scores, faith_scores) -> list[dict]:
    faith_by = {s["id"]: s for s in faith_scores}
    combined = []
    for p in primary_scores:
        f = faith_by[p["id"]]
        combined.append({"id": p["id"],
                         **{k: p[k] for k in PRIMARY_SCORE_FIELDS},
                         **{k: f[k] for k in FAITH_SCORE_FIELDS}})
    return combined


# ── 汇总 ───────────────────────────────────────────────────────────
def overall_pass(correctness: int, faithfulness: int, citation: int) -> bool:
    return correctness == 2 and faithfulness == 1 and citation == 1


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def summarize(packet, scores, mapping) -> dict:
    score_by_id = {s["id"]: s for s in scores}
    packet_by_id = {p["id"]: p for p in packet}
    per_k = {"k4": [], "k6": []}
    tele_k = {"k4": [], "k6": []}
    for _id, m in mapping.items():
        s = score_by_id[_id]; pk = packet_by_id[_id]
        for side, klabel in (("a", m["A"]), ("b", m["B"])):
            rec = {"id": _id, "correctness": s[f"{side}_correctness"],
                   "faithfulness": s[f"{side}_faithfulness"],
                   "citation": s[f"{side}_citation_correct"]}
            rec["pass"] = overall_pass(rec["correctness"], rec["faithfulness"], rec["citation"])
            per_k[klabel].append(rec)
            tele_k[klabel].append(pk[f"telemetry_{side}"])

    def agg_quality(rows):
        n = len(rows)
        return {"n": n, "overall_pass": sum(r["pass"] for r in rows),
                "overall_pass_rate": round(sum(r["pass"] for r in rows) / n, 4) if n else 0.0,
                "avg_correctness": _mean([r["correctness"] for r in rows]),
                "faithfulness_pass_rate": round(sum(r["faithfulness"] for r in rows) / n, 4) if n else 0.0,
                "citation_pass_rate": round(sum(r["citation"] for r in rows) / n, 4) if n else 0.0,
                "faithfulness_fail": sum(1 for r in rows if r["faithfulness"] == 0),
                "citation_fail": sum(1 for r in rows if r["citation"] == 0)}

    def agg_cost(teles):
        ins = [t["in_tokens"] for t in teles]; outs = [t["out_tokens"] for t in teles]
        lat = [t["latency"] for t in teles]
        return {"n": len(teles), "avg_in_tokens": _mean(ins), "avg_out_tokens": _mean(outs),
                "total_tokens": sum(ins) + sum(outs), "avg_latency": _mean(lat),
                "p50_latency": _pct(lat, 50), "p95_latency": _pct(lat, 95)}

    quality = {k: agg_quality(per_k[k]) for k in ("k4", "k6")}
    cost = {k: agg_cost(tele_k[k]) for k in ("k4", "k6")}
    p4 = {r["id"]: r["pass"] for r in per_k["k4"]}
    p6 = {r["id"]: r["pass"] for r in per_k["k6"]}
    k6_win, k4_win, tie = [], [], []
    for _id in sorted(mapping):
        a, b = p4[_id], p6[_id]
        (k6_win if (b and not a) else k4_win if (a and not b) else tie).append(_id)
    paired = {"k6_win": k6_win, "k4_win": k4_win, "tie": tie,
              "net_k6_wins": len(k6_win) - len(k4_win)}
    regressions = [_id for _id in sorted(mapping) if p4[_id] and not p6[_id]]
    faith_ok = quality["k6"]["faithfulness_fail"] <= quality["k4"]["faithfulness_fail"]
    cite_ok = quality["k6"]["citation_fail"] <= quality["k4"]["citation_fail"]
    gate = {"net_k6_wins>=2": paired["net_k6_wins"] >= 2,
            "no_overall_pass_regression": len(regressions) == 0,
            "faithfulness_fail_not_increase": faith_ok,
            "citation_fail_not_increase": cite_ok, "regression_ids": regressions}
    gate["k6_candidate"] = bool(gate["net_k6_wins>=2"] and gate["no_overall_pass_regression"]
                                and faith_ok and cite_ok)
    return {"n_questions": len(mapping), "quality": quality, "cost": cost,
            "paired": paired, "decision_gate": gate,
            "note": ("dev-only 探索，非 test 泛化；决策门槛仅报告是否满足，不代表上线；"
                     "成本/延迟上限须在真实调用前由用户明确冻结。")}


# ── IO 助手 ────────────────────────────────────────────────────────
def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_wrapped(path: Path, data_key: str):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("experiment_id"), obj.get(data_key)


def load_dev() -> list[dict]:
    return [json.loads(x) for x in DEV_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_status(out_dir, experiment_id, attempts, successes, errors, answers_ready, error_type):
    _write_json(out_dir / STATUS_FILE, {
        "experiment_id": experiment_id, "api_attempts": attempts,
        "api_successes": successes, "api_errors": errors,
        "answers_ready": answers_ready, "error_type": error_type,
        "generation_config": {"model": EVAL_MODEL, "thinking": EVAL_THINKING,
                              "max_tokens": EVAL_MAX_TOKENS,
                              "temperature": EVAL_TEMPERATURE}})


def prepare_exit(n_answers, expected=EXPECTED_ANSWERS, api_error=False, key_missing=False):
    if api_error or key_missing:
        return 2
    return 0 if n_answers >= expected else 2


# ── 模式：prepare ──────────────────────────────────────────────────
def run_prepare(args) -> int:
    out_dir = OUTPUT_DIR
    dev = load_dev()
    dev_by_id = {it["id"]: it for it in dev}
    expected_ids = [it["id"] for it in dev]
    expected = 2 * len(dev)
    scores_path = out_dir / SCORES_FILE
    faith_paths = [out_dir / FAITH_PACKET_FILE, out_dir / FAITH_SCORES_FILE]

    # 错配防护：已有主评分或进入过忠实性阶段时，无 --force 立即退出，不动任何产物。
    protected_review = (scores_path.exists()
                        and not is_pristine_scaffold(scores_path, expected_ids))
    faith_started = any(p.exists() for p in faith_paths)
    if protected_review or faith_started:
        if not args.force:
            print("✗ 主评分非空白/损坏，或 faithfulness 阶段已经开始。")
            print("  为避免评分与新 plan/mapping/packet 错配：未检索、未覆盖任何产物、非零退出。")
            print("  如确需废弃整次评测状态并重建全部产物，请显式 --force。")
            _write_status(out_dir, None, 0, 0, 0, 0, "scores_not_pristine")
            return 2
        print("⚠ --force：将废弃整次评测状态，并一起重建 plan/mapping/packet/scores（含已填评分）。")

    print(f"[prepare] dev={len(dev)} 应答={expected} "
          f"模式={'ALLOW-API' if args.allow_api else 'dry-run/cache-only'} force={args.force}")

    api_attempts = api_successes = api_errors = 0
    n_answers = 0
    experiment_id = None
    error_type = None
    rc = 2
    try:
        plan = build_plan(dev, _default_retrieve)
        assert_prefix(plan)
        mapping = make_ab_mapping(expected_ids, seed=args.seed)
        experiment_id = compute_experiment_id(dev, plan, mapping, args.seed)

        if args.allow_api:
            if not _key_ok():
                print("✗ --allow-api 但 key 无效/占位符 → 非零退出（未发起调用）")
                error_type = "key_invalid"
                return prepare_exit(0, expected, key_missing=True)
            for p in plan:                                       # 本阶段不使用
                if load_answer(out_dir, p["cache_key"], p["id"], p["k"]) is not None:
                    continue
                user = build_user(p["context"], p["question"])
                api_attempts += 1
                try:
                    gen = generate_answer(SYSTEM, user)
                except Exception as e:  # noqa: BLE001 只记稳定错误类型，不含 key/完整异常
                    print(f"✗ API 异常 [{p['id']} k{p['k']}]: api_error:{type(e).__name__}")
                    api_errors += 1; error_type = f"api_error:{type(e).__name__}"; break
                try:
                    ans, in_tok, out_tok, lat = gen
                except (TypeError, ValueError):
                    ans = in_tok = out_tok = lat = None
                # 写缓存前严格校验：只有合法返回才计成功并落盘（不打印完整答案）
                if not _valid_generation(ans, in_tok, out_tok, lat):
                    print(f"✗ 生成结果非法 [{p['id']} k{p['k']}]: output_validation_error")
                    api_errors += 1; error_type = "output_validation_error"; break
                api_successes += 1
                _write_json(_cache_dir(out_dir) / f"{p['cache_key']}.json", {
                    "id": p["id"], "k": p["k"], "cache_key": p["cache_key"],
                    "model": EVAL_MODEL, "thinking": EVAL_THINKING,
                    "max_tokens": EVAL_MAX_TOKENS, "temperature": EVAL_TEMPERATURE,
                    "prompt_version": PROMPT_VERSION, "answer": ans,
                    "telemetry": {"in_tokens": in_tok, "out_tokens": out_tok, "latency": lat}})
        else:
            print("dry-run：不创建 client、不发起任何 DeepSeek 调用（api_calls=0）")

        packet, n_answers = build_packet(plan, mapping, dev_by_id, out_dir)

        # 重建主评审：忠实性阶段和旧 summary 一并失效。
        stale_paths = [*faith_paths, out_dir / SUMMARY_FILE]
        removed = []
        for stale in stale_paths:
            if stale.exists():
                stale.unlink()
                removed.append(stale.name)
        if removed:
            print(f"已删除随主评审重建失效的产物: {removed}")

        _write_json(out_dir / PLAN_FILE, {"experiment_id": experiment_id, "plan": plan})
        _write_json(out_dir / MAPPING_FILE, {"experiment_id": experiment_id, "seed": args.seed,
                    "note": "A/B→k4/k6 真值，供解盲；评分完成前勿看", "mapping": mapping})
        _write_json(out_dir / PACKET_FILE, {"experiment_id": experiment_id, "packet": packet})
        _write_json(out_dir / SCORES_FILE, {"experiment_id": experiment_id,
                    "scores": scaffold_scores(expected_ids)})

        print(f"\napi_attempts={api_attempts} api_successes={api_successes} api_errors={api_errors}")
        print(f"experiment_id={experiment_id[:12]}…  答案就绪 {n_answers}/{expected}")
        if n_answers >= expected and api_errors == 0:
            print("✓ 答案齐备：先填写主评 review_scores.json，再运行 --prepare-faithfulness。")
        else:
            print("⚠ INCOMPLETE：答案不足或有 API 异常 —— 未就绪，不可汇总。")
        rc = prepare_exit(n_answers, expected, api_error=api_errors > 0)
        return rc
    except Exception as e:  # noqa: BLE001 记录稳定错误类型后非零退出
        error_type = error_type or f"exception:{type(e).__name__}"
        print(f"✗ prepare 异常: {type(e).__name__}")
        rc = 2
        return rc
    finally:
        _write_status(out_dir, experiment_id, api_attempts, api_successes, api_errors,
                      n_answers, error_type)


# ── 模式：主评分锁定后生成 faithfulness 审计包（不检索/不调 API）───
def run_prepare_faithfulness(args) -> int:
    out_dir = OUTPUT_DIR
    dev = load_dev()
    expected_ids = [it["id"] for it in dev]
    need = [PLAN_FILE, MAPPING_FILE, PACKET_FILE, SCORES_FILE]
    missing = [f for f in need if not (out_dir / f).exists()]
    if missing:
        print(f"✗ 缺少主评审文件: {missing} —— 无法进入 faithfulness 阶段。")
        return 2

    faith_scores_path = out_dir / FAITH_SCORES_FILE
    if faith_scores_path.exists() and not is_pristine_scaffold(
            faith_scores_path, expected_ids, FAITH_SCORE_FIELDS, FAITH_SCAFFOLD_KEYS):
        print("✗ faithfulness_scores.json 已填写/损坏；拒绝覆盖。")
        return 2

    try:
        plan_eid, plan = _read_wrapped(out_dir / PLAN_FILE, "plan")
        map_eid, mapping = _read_wrapped(out_dir / MAPPING_FILE, "mapping")
        pkt_eid, packet = _read_wrapped(out_dir / PACKET_FILE, "packet")
        sc_eid, scores = _read_wrapped(out_dir / SCORES_FILE, "scores")
        seed = json.loads((out_dir / MAPPING_FILE).read_text(encoding="utf-8")).get("seed")
    except (OSError, ValueError) as e:
        print(f"✗ 读取主评审产物失败: {type(e).__name__}")
        return 2

    eids = {plan_eid, map_eid, pkt_eid, sc_eid}
    if None in eids or len(eids) != 1:
        print("✗ 主评审 experiment_id 不一致或缺失。")
        return 2

    errors = []
    errors += validate_mapping(mapping, expected_ids)
    errors += validate_packet(packet, dev)
    errors += validate_scores(scores, expected_ids)
    errors += validate_plan(plan, dev, mapping)
    if not errors:
        errors += validate_packet_derived(packet, dev, plan, mapping)
    if not errors and compute_experiment_id(dev, plan, mapping, seed) != plan_eid:
        errors.append("experiment_id 重算不一致")
    if errors:
        print("✗ 主评分未完整或主评审产物校验失败，不能生成 faithfulness packet：")
        for e in errors:
            print("   -", e)
        return 2

    faith_packet = build_faithfulness_packet(plan, mapping, out_dir)
    errors = validate_faithfulness_packet(faith_packet, dev)
    errors += validate_faithfulness_packet_derived(faith_packet, dev, plan, mapping)
    errors += validate_packets_agree(packet, faith_packet)
    if errors:
        print("✗ faithfulness packet 构造校验失败：")
        for e in errors:
            print("   -", e)
        return 2

    locked_hash = primary_scores_hash(scores)
    wrapper_meta = {"experiment_id": plan_eid, "primary_scores_hash": locked_hash}
    _write_json(out_dir / FAITH_PACKET_FILE, {**wrapper_meta, "packet": faith_packet})
    _write_json(faith_scores_path, {**wrapper_meta,
                "scores": scaffold_scores(expected_ids, FAITH_SCORE_FIELDS)})
    stale_summary = out_dir / SUMMARY_FILE
    if stale_summary.exists():
        stale_summary.unlink()
    print(f"✓ 主评分已锁定 hash={locked_hash[:12]}…；生成 faithfulness 审计包与评分脚手架。")
    print("  现在只填写 faithfulness_scores.json；不要修改 review_scores.json。")
    return 0


# ── 模式：summarize（只读；不检索/不建 client/不调 API）─────────────
def run_summarize(args) -> int:
    out_dir = OUTPUT_DIR
    dev = load_dev()
    expected_ids = [it["id"] for it in dev]

    need = [PLAN_FILE, MAPPING_FILE, PACKET_FILE, SCORES_FILE,
            FAITH_PACKET_FILE, FAITH_SCORES_FILE]
    missing = [f for f in need if not (out_dir / f).exists()]
    if missing:
        print(f"✗ 缺少必要文件: {missing} —— 无法汇总。")
        return 2

    try:
        plan_eid, plan = _read_wrapped(out_dir / PLAN_FILE, "plan")
        map_eid, mapping = _read_wrapped(out_dir / MAPPING_FILE, "mapping")
        pkt_eid, packet = _read_wrapped(out_dir / PACKET_FILE, "packet")
        sc_eid, scores = _read_wrapped(out_dir / SCORES_FILE, "scores")
        faith_pkt_obj = json.loads((out_dir / FAITH_PACKET_FILE).read_text(encoding="utf-8"))
        faith_sc_obj = json.loads((out_dir / FAITH_SCORES_FILE).read_text(encoding="utf-8"))
        faith_pkt_eid, faith_packet = faith_pkt_obj.get("experiment_id"), faith_pkt_obj.get("packet")
        faith_sc_eid, faith_scores = faith_sc_obj.get("experiment_id"), faith_sc_obj.get("scores")
        seed = json.loads((out_dir / MAPPING_FILE).read_text(encoding="utf-8")).get("seed")
    except (OSError, ValueError) as e:
        print(f"✗ 读取产物失败: {type(e).__name__}")
        return 2

    eids = {plan_eid, map_eid, pkt_eid, sc_eid, faith_pkt_eid, faith_sc_eid}
    if None in eids or len(eids) != 1:
        print("✗ 两阶段 experiment_id 不一致或缺失 —— 拒绝汇总。")
        return 2

    locked_hash = primary_scores_hash(scores)
    if (faith_pkt_obj.get("primary_scores_hash") != locked_hash
            or faith_sc_obj.get("primary_scores_hash") != locked_hash):
        print("✗ 主评分在 faithfulness 阶段生成后被修改，或锁定 hash 缺失 —— 拒绝汇总。")
        return 2

    # 结构校验
    errors = []
    errors += validate_mapping(mapping, expected_ids)
    errors += validate_packet(packet, dev)
    errors += validate_scores(scores, expected_ids)
    errors += validate_faithfulness_packet(faith_packet, dev)
    errors += validate_faithfulness_scores(faith_scores, expected_ids)
    # 深度内容校验：plan 内容、packet 派生字段
    errors += validate_plan(plan, dev, mapping)
    if not errors:                                    # 派生校验依赖 plan/mapping 结构先成立
        errors += validate_packet_derived(packet, dev, plan, mapping)
        errors += validate_faithfulness_packet_derived(faith_packet, dev, plan, mapping)
        errors += validate_packets_agree(packet, faith_packet)
        if build_faithfulness_packet(plan, mapping, out_dir) != faith_packet:
            errors.append("faithfulness packet 与当前严格缓存不一致")
    if errors:
        print("✗ 汇总前校验失败，不生成 summary：")
        for e in errors:
            print("   -", e)
        return 2

    # 用加载出的 dev/plan/mapping/seed 重算 experiment_id，须等于四文件声明的 ID
    recomputed = compute_experiment_id(dev, plan, mapping, seed)
    if recomputed != plan_eid:
        print(f"✗ experiment_id 重算不一致（声明 {plan_eid[:12]}… ≠ 重算 {recomputed[:12]}…）—— 拒绝汇总。")
        return 2

    telemetry_packet = build_telemetry_packet(plan, mapping, out_dir)
    telemetry_errors = validate_telemetry_packet(telemetry_packet, expected_ids)
    if telemetry_errors:
        print("✗ telemetry 缓存校验失败：")
        for e in telemetry_errors:
            print("   -", e)
        return 2
    summary = summarize(telemetry_packet, combine_scores(scores, faith_scores), mapping)
    # 成本指标要求 k4/k6 各 24 条合法 telemetry（validate_packet 已保证，此处防御）
    if summary["cost"]["k4"]["n"] != len(expected_ids) or summary["cost"]["k6"]["n"] != len(expected_ids):
        print("✗ telemetry 数量不足，无法生成成本指标。")
        return 2
    summary["experiment_id"] = plan_eid
    _write_json(out_dir / SUMMARY_FILE, summary)
    q = summary["quality"]
    print(f"[summarize] eid={plan_eid[:12]}…  k4 pass {q['k4']['overall_pass']}/{q['k4']['n']}  "
          f"k6 pass {q['k6']['overall_pass']}/{q['k6']['n']}  "
          f"net_k6_wins={summary['paired']['net_k6_wins']}  "
          f"k6_candidate={summary['decision_gate']['k6_candidate']}")
    print(f"✓ 生成 {out_dir/SUMMARY_FILE}")
    return 0


def _default_retrieve(question: str) -> list[dict]:
    from app import hybrid                       # 惰性导入，避免模块导入即连 Chroma
    return hybrid.hybrid_search(question, k=K_BIG)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true", help="只读评分与 mapping 解盲汇总（不检索/不调 API）")
    ap.add_argument("--prepare-faithfulness", action="store_true",
                    help="主评分完整后生成忠实性审计包（不检索/不调 API）")
    ap.add_argument("--allow-api", action="store_true", help="prepare 时允许真实生成（默认 dry-run）")
    ap.add_argument("--force", action="store_true", help="废弃整次评测状态并重建全部产物")
    ap.add_argument("--seed", type=int, default=AB_SEED, help="A/B 映射随机种子（默认固定）")
    args = ap.parse_args()
    special_modes = int(args.summarize) + int(args.prepare_faithfulness)
    if special_modes > 1 or ((args.summarize or args.prepare_faithfulness)
                             and (args.allow_api or args.force)):
        print("✗ --summarize / --prepare-faithfulness 互斥，且不能与 --allow-api / --force 同用。")
        return 2
    if args.summarize:
        return run_summarize(args)
    if args.prepare_faithfulness:
        return run_prepare_faithfulness(args)
    return run_prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""方案 A 检索流水线：大候选池 → 同源相邻扩展 → 去重 → rerank/筛选 → 有上限截断。

设计成可组合、可注入依赖，便于离线单测：核心排序/扩展/截断都是纯函数，
不在其中触网或读库；只有顶层 run() 会去调用 hybrid / embeddings / vectorstore。

不修改 hybrid.py / rag.py，也不改变正式 /ask 的默认行为——本模块仅供实验使用。
"""
import numpy as np

from . import hybrid, vectorstore, embeddings

RRF_C = 60          # 与既有 hybrid 一致的 RRF 常数
DECAY = 0.9         # A0 邻居继承衰减


# ── Stage1+2：同源相邻扩展（内含按 id 去重）─────────────────────────
def expand_adjacency(anchors: list[dict], lookup: dict, w: int) -> list[dict]:
    """anchors 按融合分降序；每个锚点 (source, idx) 扩展同源邻居 idx-w..idx+w。

    lookup: dict[(source, chunk_index)] -> chunk({id,text,source,chunk_index})。
    邻居继承锚点分 × DECAY**距离；被多个锚点覆盖时取 max。返回**去重后**的候选池。
    """
    cand: dict[str, dict] = {}
    for a in anchors:
        src, idx = a["source"], a["chunk_index"]
        for d in range(-w, w + 1):
            ch = lookup.get((src, idx + d))     # 越界的邻居自然缺失 → 跳过（clamp）
            if ch is None:
                continue
            inherited = a["score"] * (DECAY ** abs(d))
            cur = cand.get(ch["id"])
            if cur is None:
                cand[ch["id"]] = {**ch, "inherited": inherited, "is_anchor": d == 0}
            else:
                cur["inherited"] = max(cur["inherited"], inherited)
                cur["is_anchor"] = cur["is_anchor"] or d == 0
    return list(cand.values())


# ── Stage3：三种排序（A0 / A-b / A-a）──────────────────────────────
def _cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _by(pool, score_of):
    # 降序；平手用 (source, chunk_index) 保证确定性
    return sorted(pool, key=lambda x: (-score_of(x), x["source"], x["chunk_index"]))


def order_a0(pool: list[dict]) -> list[dict]:
    """仅按继承分排序（无需 embedding，最简单、确定）。"""
    return _by(pool, lambda x: x["inherited"])


def _rrf_of_two(pool, order1, order2):
    rrf: dict[str, float] = {}
    for r, x in enumerate(order1):
        rrf[x["id"]] = rrf.get(x["id"], 0.0) + 1.0 / (RRF_C + r)
    for r, x in enumerate(order2):
        rrf[x["id"]] = rrf.get(x["id"], 0.0) + 1.0 / (RRF_C + r)
    return _by(pool, lambda x: rrf[x["id"]])


def order_ab(pool: list[dict], query_vec, emb_lookup: dict) -> list[dict]:
    """继承名次 ⊕ cosine 名次，用 RRF 二次融合（尺度无关、无需归一化）。"""
    cos = {x["id"]: _cosine(query_vec, emb_lookup[x["id"]]) for x in pool}
    inh_order = _by(pool, lambda x: x["inherited"])
    cos_order = _by(pool, lambda x: cos[x["id"]])
    return _rrf_of_two(pool, inh_order, cos_order)


def order_aa(pool: list[dict], query_vec, emb_lookup: dict, alpha: float = 0.5) -> list[dict]:
    """归一化加权（仅作单点参照：min-max 归一后 α·继承 + (1-α)·cosine）。"""
    inh = {x["id"]: x["inherited"] for x in pool}
    cos = {x["id"]: _cosine(query_vec, emb_lookup[x["id"]]) for x in pool}

    def norm(d):
        vs = list(d.values())
        lo, hi = min(vs), max(vs)
        return {k: (0.0 if hi == lo else (v - lo) / (hi - lo)) for k, v in d.items()}

    ninh, ncos = norm(inh), norm(cos)
    final = {x["id"]: alpha * ninh[x["id"]] + (1 - alpha) * ncos[x["id"]] for x in pool}
    return _by(pool, lambda x: final[x["id"]])


# ── Stage4：每源限流 + 全局上限 ────────────────────────────────────
def cap(ordered: list[dict], per_source: int, k_final: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for x in ordered:
        if seen.get(x["source"], 0) >= per_source:
            continue
        seen[x["source"]] = seen.get(x["source"], 0) + 1
        out.append(x)
        if len(out) >= k_final:
            break
    return out


# ── 顶层：组合整条流水线（实验用）─────────────────────────────────
def build_lookup() -> dict:
    return {(c["source"], c["chunk_index"]): c for c in vectorstore.get_all()}


def run(question: str, k_final: int, w: int, fusion: str = "A0",
        p: int = 12, s: int = 3, pool_top_n: int = 20, alpha: float = 0.5,
        query_vec=None, emb_lookup: dict | None = None, lookup: dict | None = None) -> dict:
    """返回 {anchors, pool, final}，供实验脚本分阶段度量。"""
    if fusion not in ("A0", "A-b", "A-a"):     # 尽早报错：检索/embedding 之前
        raise ValueError(f"unknown fusion: {fusion!r}（应为 'A0'/'A-b'/'A-a'）")
    anchors = hybrid.hybrid_search(question, k=p, top_n=pool_top_n)
    lookup = lookup if lookup is not None else build_lookup()
    pool = expand_adjacency(anchors, lookup, w)

    if fusion == "A0":
        ordered = order_a0(pool)
    else:
        if emb_lookup is None:
            emb_lookup = vectorstore.get_embeddings()
        if query_vec is None:
            query_vec = embeddings.embed_query(question)
        ordered = order_ab(pool, query_vec, emb_lookup) if fusion == "A-b" \
            else order_aa(pool, query_vec, emb_lookup, alpha)

    return {"anchors": anchors, "pool": pool, "final": cap(ordered, s, k_final)}


# ══════════════════════════════════════════════════════════════════════
# 窗口级排序（下一轮实验）：window 为不可分割的选择单位。
# 整窗按"窗内最高分成员"打分——锚点或任一邻居得分高，即把整窗（连同证据邻居）带入。
# ══════════════════════════════════════════════════════════════════════
def make_windows(anchors: list[dict], lookup: dict, w: int) -> list[dict]:
    """每个锚点 → 一个 window = 同源 {idx-w..idx+w}（clamp）。窗口是原子单位。"""
    wins = []
    for rank, a in enumerate(anchors):
        src, idx = a["source"], a["chunk_index"]
        chunks = [lookup[(src, idx + d)] for d in range(-w, w + 1) if (src, idx + d) in lookup]
        wins.append({"anchor_source": src, "anchor_idx": idx, "anchor_rank": rank,
                     "anchor_score": a["score"], "chunks": chunks,
                     "chunk_ids": [c["id"] for c in chunks]})
    return wins


def score_windows(windows: list[dict], query_vec, emb_lookup: dict,
                  method: str = "WS-max") -> list[dict]:
    """给每个 window 打分并返回降序列表。窗分聚合窗内 chunk 的 query-cosine。"""
    if method not in ("WS-max", "WS-mean", "WS-anchor+max"):   # 尽早报错：修改 windows 之前
        raise ValueError(f"unknown method: {method!r}（应为 'WS-max'/'WS-mean'/'WS-anchor+max'）")
    for win in windows:
        coss = [_cosine(query_vec, emb_lookup[c["id"]]) for c in win["chunks"]]
        win["score"] = (sum(coss) / len(coss)) if method == "WS-mean" else max(coss)

    def _tie(win):
        return (win["anchor_source"], win["anchor_idx"])

    if method == "WS-anchor+max":
        by_anchor = sorted(windows, key=lambda w: (w["anchor_rank"], *_tie(w)))
        by_cos = sorted(windows, key=lambda w: (-w["score"], *_tie(w)))
        rrf = {}
        for r, win in enumerate(by_anchor):
            rrf[id(win)] = rrf.get(id(win), 0.0) + 1.0 / (RRF_C + r)
        for r, win in enumerate(by_cos):
            rrf[id(win)] = rrf.get(id(win), 0.0) + 1.0 / (RRF_C + r)
        for win in windows:
            win["fused"] = rrf[id(win)]
        return sorted(windows, key=lambda w: (-w["fused"], *_tie(w)))
    return sorted(windows, key=lambda w: (-w["score"], *_tie(w)))


def select_windows(ordered_windows: list[dict], k_final: int, s_per_source: int) -> list[dict]:
    """原子选窗：整窗加入若使 distinct chunk 超过 K_final → **跳过整窗**，继续看后面的窗。

    不在 K_final 边界部分保留窗口，避免证据邻居被裁。每源最多 s_per_source 个窗。
    """
    selected, chosen_ids, per_source = [], set(), {}
    for win in ordered_windows:
        src = win["anchor_source"]
        if per_source.get(src, 0) >= s_per_source:
            continue
        new_ids = [cid for cid in win["chunk_ids"] if cid not in chosen_ids]
        if not new_ids:
            continue                                   # 完全重叠，不占名额
        if len(chosen_ids) + len(new_ids) > k_final:   # 原子：会超预算 → 跳过整窗
            continue
        selected.append(win)
        per_source[src] = per_source.get(src, 0) + 1
        chosen_ids.update(win["chunk_ids"])
    return selected


def dedup_order(selected: list[dict]) -> list[dict]:
    """选中窗 → 展开 → **去重 + 排序**（顺序=选窗顺序，窗内按 chunk_index）。

    注意：本函数只做 dedup + 排序，**不**把同源相邻 index 合并成连续文本段。
    （前称 dedup_order，名称有误导，已更正。）
    """
    final, seen = [], set()
    for win in selected:
        for c in sorted(win["chunks"], key=lambda x: x["chunk_index"]):
            if c["id"] not in seen:
                seen.add(c["id"])
                final.append(c)
    return final


def run_window(question: str, k_final: int, w: int, method: str = "WS-max",
               p: int = 12, s: int = 3, pool_top_n: int = 20,
               query_vec=None, emb_lookup: dict | None = None, lookup: dict | None = None) -> dict:
    anchors = hybrid.hybrid_search(question, k=p, top_n=pool_top_n)
    lookup = lookup if lookup is not None else build_lookup()
    if emb_lookup is None:
        emb_lookup = vectorstore.get_embeddings()
    if query_vec is None:
        query_vec = embeddings.embed_query(question)
    windows = make_windows(anchors, lookup, w)
    ordered = score_windows(windows, query_vec, emb_lookup, method)
    selected = select_windows(ordered, k_final, s)
    pool_ids = {cid for win in windows for cid in win["chunk_ids"]}
    return {"anchors": anchors, "windows": windows, "n_windows": len(windows),
            "pool_size": len(pool_ids), "selected": selected,
            "final": dedup_order(selected)}


# ══════════════════════════════════════════════════════════════════════
# 基线保留式补充检索（第一阶段：完全确定、无 rerank）
#
# 原则：hybrid top-K 作为「基础上下文」，内容与顺序**完全冻结**；补充邻居只能
# **追加在末尾**，绝不删除或重排 base。因此最坏情况 = 退回基线，不会变差。
#
# 原子单元：候选 = (anchor, neighbor)。
#   - anchor 不在 base 中 → 原子单元 = [anchor, neighbor]（两块一起加或一起跳）
#   - anchor 已在 base 中 → 原子单元 = [neighbor]（只补邻居）
# 单元不可拆：其新增 chunk 的 token 放不进该题剩余预算 → 跳过整单元，继续后续候选。
# 预算：补充 token ≤ 0.5 × 该题 base token，使总 ≤ 1.5 × base。
# ══════════════════════════════════════════════════════════════════════
def _toks(text: str) -> int:
    return len(text) // 4


def build_candidates(base: list[dict], anchors: list[dict], lookup: dict, mode: str) -> list[dict]:
    """生成补充候选原子单元。

    复合 unit_id = anchor_chunk_id + "->" + neighbor_chunk_id（同一 neighbor 可能来自
    不同 anchor，成本/单元不同，故必须用复合 id 唯一标识）。
    - anchor 不在 base → unit = [anchor, neighbor]
    - anchor 已在 base → unit = [neighbor]
    """
    if mode == "N-next":
        deltas = [1]
    elif mode == "N-win":
        deltas = [-1, 1]
    else:
        raise ValueError(f"unknown mode: {mode!r}（应为 'N-next' 或 'N-win'）")
    base_ids = {c["id"] for c in base}
    cands, seen_uid = [], set()
    for rank, a in enumerate(anchors):
        src, idx = a["source"], a["chunk_index"]
        anchor_chunk = lookup.get((src, idx))
        anchor_id = a.get("id") or (anchor_chunk and anchor_chunk["id"])
        for d in deltas:
            nb = lookup.get((src, idx + d))
            if nb is None:
                continue                       # 越界邻居 clamp
            uid = f"{anchor_id}->{nb['id']}"
            if uid in seen_uid:
                continue
            seen_uid.add(uid)
            unit = [nb] if anchor_id in base_ids else ([anchor_chunk, nb] if anchor_chunk else [nb])
            cands.append({
                "unit_id": uid, "anchor_id": anchor_id, "neighbor_id": nb["id"],
                "anchor_rank": rank, "dist": abs(d), "delta": d,
                "nb_idx": nb["chunk_index"], "source": src,
                "anchor_text": (anchor_chunk or {}).get("text", ""),
                "neighbor_text": nb["text"], "unit": unit,
            })
    return cands


def _default_order_key(c):
    # 确定性 tie 规则：anchor hybrid 名次 → 邻居距离 |delta| → 邻居 chunk_index → source
    return (c["anchor_rank"], c["dist"], c["nb_idx"], c["source"])


def supplement_base(base: list[dict], anchors: list[dict], lookup: dict,
                    mode: str, budget_factor: float = 0.5,
                    order_unit_ids: list[str] | None = None) -> dict:
    """在冻结的 base 之后追加补充邻居。

    order_unit_ids: 若给定（rerank 返回的 unit_id 全排列），补充单元按它排序；否则用
      确定性默认序。base top6 内容与顺序完全冻结，补充只 append。
    """
    base_ids = {c["id"] for c in base}
    base_tok = sum(_toks(c["text"]) for c in base)
    budget = budget_factor * base_tok
    cands = build_candidates(base, anchors, lookup, mode)

    if order_unit_ids is not None:
        pos = {uid: i for i, uid in enumerate(order_unit_ids)}
        cands.sort(key=lambda c: (pos.get(c["unit_id"], len(order_unit_ids)), _default_order_key(c)))
    else:
        cands.sort(key=_default_order_key)

    # 预算内逐单元加入（原子，不可拆）
    added, added_ids, remaining = [], set(), budget
    for c in cands:
        new_chunks = [ch for ch in c["unit"]
                      if ch["id"] not in base_ids and ch["id"] not in added_ids]
        if not new_chunks:
            continue                           # 完全重叠
        cost = sum(_toks(ch["text"]) for ch in new_chunks)
        if cost > remaining:
            continue                           # 放不下整单元 → 跳过，继续后续
        for ch in sorted(new_chunks, key=lambda x: x["chunk_index"]):
            added.append(ch)
            added_ids.add(ch["id"])
        remaining -= cost

    final = list(base) + added                 # base 冻结在前，补充追加在后
    return {"base": base, "supplements": added, "final": final,
            "base_tokens": base_tok, "supp_tokens": budget - remaining,
            "candidates": cands, "n_candidates": len(cands)}


def run_supplement(question: str, k_base: int, mode: str, p: int = 12,
                   budget_factor: float = 0.5, pool_top_n: int = 20,
                   lookup: dict | None = None, order_unit_ids: list[str] | None = None) -> dict:
    base = hybrid.hybrid_search(question, k=k_base)
    anchors = hybrid.hybrid_search(question, k=p, top_n=pool_top_n)
    lookup = lookup if lookup is not None else build_lookup()
    r = supplement_base(base, anchors, lookup, mode, budget_factor, order_unit_ids)
    r["anchors"] = anchors
    return r

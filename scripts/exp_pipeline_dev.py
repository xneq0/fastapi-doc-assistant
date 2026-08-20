"""方案 A 的 dev-only 实验（R0→R1→R2→R3）。不改 /ask 默认行为，仅评测。

护栏：G1 去重后平均池 ≤ P*2(=24)；G2 最终平均 token ≤ 1.5× 同 K_final 原始 hybrid 基线。
硬门槛：dev20 的 path-params.md#2 进入最终上下文。
只用 dev，不跑 test。运行：./.venv/bin/python -m scripts.exp_pipeline_dev
"""
import json
from pathlib import Path
from statistics import mean

from app import vectorstore, hybrid, ingest, embeddings, retrieval_pipeline as rp
from app.eval_metrics import source_rank, evidence_chunk_rank, reciprocal_rank

DEV = [json.loads(x) for x in Path("eval/dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
P, S = 12, 3                 # P=12 锚点，per-source cap S=3
G1_MAX_POOL = P * 2          # 24
G2_FACTOR = 1.5
DEV20 = next(it for it in DEV if it["id"] == "dev20")


def toks(chunks):
    return sum(len(c["text"]) // 4 for c in chunks)


def ev_rank(lst, it):
    return evidence_chunk_rank(lst, it["expected_source"], it.get("evidence_quote"))


def agg_final(finals):
    evs = [ev_rank(f, it) for f, it in zip(finals, DEV)]
    srs = [source_rank(f, it["expected_source"]) for f, it in zip(finals, DEV)]
    return {
        "ev_recall": mean(r is not None for r in evs),
        "ev_mrr": mean(reciprocal_rank(r) for r in evs),
        "src_recall": mean(r is not None for r in srs),
        "avg_chunks": mean(len(f) for f in finals),
        "avg_tokens": mean(toks(f) for f in finals),
    }


def dev20_pass(final_list):
    return any(c["source"] == "tutorial/path-params.md" and c["chunk_index"] == 2 for c in final_list)


# ── 预备：干净重建索引 + 预算 query 向量 / 库内向量 / 邻居表 ──────────
def setup():
    print("重建索引：reset + 从 docs/ 入库 ...")
    vectorstore.reset()
    ingest.ingest_dir("docs")
    print("库内片段:", vectorstore.count())
    qs = [it["question"] for it in DEV]
    qvec = dict(zip(qs, embeddings.embed_texts(qs)))
    return qvec, vectorstore.get_embeddings(), rp.build_lookup()


def baseline(K):
    return [hybrid.hybrid_search(it["question"], k=K) for it in DEV]


def pipeline(K, w, fusion, ctx):
    qvec, emb, lookup = ctx
    finals, pools = [], []
    for it in DEV:
        r = rp.run(it["question"], k_final=K, w=w, fusion=fusion, p=P, s=S,
                   query_vec=None if fusion == "A0" else qvec[it["question"]],
                   emb_lookup=emb, lookup=lookup)
        finals.append(r["final"]); pools.append(r["pool"])
    return finals, pools


def net_change(a_finals, base_finals):
    fixed, broken = [], []
    for it, af, bf in zip(DEV, a_finals, base_finals):
        a_hit = ev_rank(af, it) is not None
        b_hit = ev_rank(bf, it) is not None
        if a_hit and not b_hit:
            fixed.append(it["id"])
        elif b_hit and not a_hit:
            broken.append(it["id"])
    return fixed, broken


def line(name, m):
    return (f"  {name:<10} ev-recall={m['ev_recall']:.0%}  ev-MRR={m['ev_mrr']:.3f}  "
            f"src-recall={m['src_recall']:.0%}  chunks={m['avg_chunks']:.1f}  tok={m['avg_tokens']:.0f}")


def main():
    ctx = setup()
    base_tok = {}   # K -> baseline avg tokens (G2 参照)
    base_m = {}

    # ── R0 基线 ──
    print("\n===== R0 基线（同 K_final 参照）=====")
    for K in (4, 6):
        bf = baseline(K)
        m = agg_final(bf)
        base_tok[K], base_m[K] = m["avg_tokens"], m
        print(f"B@{K}"); print(line(f"hybrid@{K}", m))

    # ── R1 定 w（A0, K=6）──
    print("\n===== R1 定窗口 w（A0, K_final=6）=====")
    r1 = {}
    for w in (1, 2):
        fin, pool = pipeline(6, w, "A0", ctx)
        m = agg_final(fin); avg_pool = mean(len(p) for p in pool)
        pool_ev = mean(ev_rank(p, it) is not None for p, it in zip(pool, DEV))
        g1 = avg_pool <= G1_MAX_POOL
        g2 = m["avg_tokens"] <= G2_FACTOR * base_tok[6]
        gate = dev20_pass(fin[DEV.index(DEV20)])
        r1[w] = dict(m=m, avg_pool=avg_pool, pool_ev=pool_ev, g1=g1, g2=g2, gate=gate, fin=fin)
        print(f"w={w}"); print(line(f"A0/w{w}", m))
        print(f"    pool-ev-recall={pool_ev:.0%}  avg_pool={avg_pool:.1f}  "
              f"G1(≤{G1_MAX_POOL})={'✓' if g1 else '✗'}  "
              f"G2(≤{G2_FACTOR}×{base_tok[6]:.0f})={'✓' if g2 else '✗'}  dev20={'✓' if gate else '✗'}")

    passing_w = [w for w in (1, 2) if r1[w]["g1"] and r1[w]["g2"] and r1[w]["gate"]]
    w_star = None
    if passing_w:
        w_star = max(passing_w, key=lambda w: (r1[w]["pool_ev"], r1[w]["m"]["ev_recall"], -w))
    print(f"→ 通过护栏+门槛的 w: {passing_w or '无'} ; w* = {w_star}")

    if w_star is None:
        print("\n✗ 阶段① 无 w 同时通过 G1/G2/门槛 —— 如实报告失败，停止。")
        return

    # ── R2 定融合（w*, K=6）；A0/A-b 主选，A-a 单点参照 ──
    print(f"\n===== R2 定融合（w={w_star}, K_final=6）=====")
    r2 = {}
    for fusion in ("A0", "A-b", "A-a"):
        fin, pool = pipeline(6, w_star, fusion, ctx)
        m = agg_final(fin)
        g2 = m["avg_tokens"] <= G2_FACTOR * base_tok[6]
        gate = dev20_pass(fin[DEV.index(DEV20)])
        r2[fusion] = dict(m=m, g2=g2, gate=gate, fin=fin)
        tag = fusion + ("（参照）" if fusion == "A-a" else "")
        print(line(tag, m) + f"   G2={'✓' if g2 else '✗'}  dev20={'✓' if gate else '✗'}")

    cands = [f for f in ("A0", "A-b") if r2[f]["g2"] and r2[f]["gate"]]
    fusion_star = None
    if cands:
        fusion_star = max(cands, key=lambda f: (r2[f]["m"]["ev_recall"], r2[f]["m"]["ev_mrr"],
                                                1 if f == "A0" else 0))
    print(f"→ fusion*（A0/A-b 中）= {fusion_star}")
    if fusion_star is None:
        print("\n✗ 阶段② A0/A-b 均未过门槛/护栏 —— 如实报告失败，停止。")
        return

    # ── R3 K_final 敏感度 + 同 K 正面对比 ──
    print(f"\n===== R3 K_final 敏感度（w={w_star}, fusion={fusion_star}）=====")
    r3 = {}
    for K in (4, 6):
        fin, pool = pipeline(K, w_star, fusion_star, ctx)
        m = agg_final(fin); avg_pool = mean(len(p) for p in pool)
        g1 = avg_pool <= G1_MAX_POOL
        g2 = m["avg_tokens"] <= G2_FACTOR * base_tok[K]
        gate = dev20_pass(fin[DEV.index(DEV20)])
        fixed, broken = net_change(fin, baseline(K))
        gain = m["ev_recall"] - base_m[K]["ev_recall"]
        mrr_ok = m["ev_mrr"] >= base_m[K]["ev_mrr"] - 1e-9
        r3[K] = dict(m=m, g1=g1, g2=g2, gate=gate, fixed=fixed, broken=broken, gain=gain, mrr_ok=mrr_ok)
        print(f"\nK_final={K}  (基线 B@{K}: ev-recall={base_m[K]['ev_recall']:.0%}, "
              f"ev-MRR={base_m[K]['ev_mrr']:.3f}, tok={base_tok[K]:.0f})")
        print(line(f"A/{fusion_star}", m))
        print(f"    Δev-recall={gain:+.0%}  MRR不劣={'✓' if mrr_ok else '✗'}  "
              f"G1={'✓' if g1 else '✗'}  G2={'✓' if g2 else '✗'}  dev20={'✓' if gate else '✗'}")
        print(f"    逐题净变化：修好{len(fixed)}题{fixed}  弄坏{len(broken)}题{broken}  净={len(fixed)-len(broken):+d}")

    # ── 全局验收 ──
    print("\n===== 全局验收 =====")
    accepted = []
    for K in (4, 6):
        r = r3[K]
        ok = (r["gate"] and r["g1"] and r["g2"] and r["gain"] > 0 and r["mrr_ok"]
              and (len(r["fixed"]) - len(r["broken"])) >= 0)
        print(f"  (w={w_star}, {fusion_star}, K={K}) 全部条件通过 = {'✓ PASS' if ok else '✗ FAIL'}")
        if ok:
            accepted.append(K)
    if accepted:
        Kstar = min(accepted)   # 拐点：取满足全部条件的最小 K（上下文更省）
        print(f"\n✓ 验收通过。推荐配置：w={w_star}, fusion={fusion_star}, K_final={Kstar}（dev-only）")
        print("  注：DeepSeek 是否正确引用 #2 另做人工回答复核，不计入本离线验收。")
    else:
        print("\n✗ 无配置同时通过硬门槛与 G1/G2/增益 —— 如实报告失败，不放宽标准。")


if __name__ == "__main__":
    main()

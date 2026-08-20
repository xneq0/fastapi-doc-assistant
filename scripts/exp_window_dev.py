"""窗口级排序的 dev-only 实验。window 为不可分割选择单位。

本轮取消 G1（池大小）硬门槛，只把候选窗口数/去重池大小作为报告指标。
保留：同 K 基线、G2 token 护栏（超标即判失败，不拆窗）、dev20 硬门槛、
final evidence-recall≥基线、ev-MRR 不劣、逐题净变化≥0。
S=3；只跑 dev；不跑 test。运行：./.venv/bin/python -m scripts.exp_window_dev
"""
import json
from pathlib import Path
from statistics import mean

from app import vectorstore, hybrid, ingest, embeddings, retrieval_pipeline as rp
from app.eval_metrics import source_rank, evidence_chunk_rank, reciprocal_rank

DEV = [json.loads(x) for x in Path("eval/dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
P, S, W = 12, 3, 1
G2_FACTOR = 1.5
DEV20 = next(it for it in DEV if it["id"] == "dev20")


def toks(chunks):
    return sum(len(c["text"]) // 4 for c in chunks)


def evr(lst, it):
    return evidence_chunk_rank(lst, it["expected_source"], it.get("evidence_quote"))


def dev20_pass(final):
    return any(c["source"] == "tutorial/path-params.md" and c["chunk_index"] == 2 for c in final)


def setup():
    print("重建索引：reset + 从 docs/ 入库 ...")
    vectorstore.reset(); ingest.ingest_dir("docs")
    print("库内片段:", vectorstore.count())
    qs = [it["question"] for it in DEV]
    return dict(zip(qs, embeddings.embed_texts(qs))), vectorstore.get_embeddings(), rp.build_lookup()


def baseline(K):
    return [hybrid.hybrid_search(it["question"], k=K) for it in DEV]


def agg(finals):
    evs = [evr(f, it) for f, it in zip(finals, DEV)]
    srs = [source_rank(f, it["expected_source"]) for f, it in zip(finals, DEV)]
    return dict(ev_recall=mean(r is not None for r in evs),
                ev_mrr=mean(reciprocal_rank(r) for r in evs),
                src_recall=mean(r is not None for r in srs),
                avg_chunks=mean(len(f) for f in finals),
                avg_tokens=mean(toks(f) for f in finals))


def net_change(a_finals, base_finals):
    fixed = broken = 0; brk = []
    for it, af, bf in zip(DEV, a_finals, base_finals):
        ah, bh = evr(af, it) is not None, evr(bf, it) is not None
        if ah and not bh:
            fixed += 1
        elif bh and not ah:
            broken += 1; brk.append(it["id"])
    return fixed, broken, brk


def line(tag, m):
    return (f"  {tag:<16} ev-recall={m['ev_recall']:.0%}  ev-MRR={m['ev_mrr']:.3f}  "
            f"src-recall={m['src_recall']:.0%}  chunks={m['avg_chunks']:.1f}  tok={m['avg_tokens']:.0f}")


def run_method(method, K, ctx):
    qvec, emb, lookup = ctx
    finals, pool_cov, nwin, psize = [], [], [], []
    for it in DEV:
        r = rp.run_window(it["question"], k_final=K, w=W, method=method, p=P, s=S,
                          query_vec=qvec[it["question"]], emb_lookup=emb, lookup=lookup)
        finals.append(r["final"]); nwin.append(r["n_windows"]); psize.append(r["pool_size"])
        allchunks = [c for win in r["windows"] for c in win["chunks"]]
        pool_cov.append(evr(allchunks, it) is not None)
    return finals, dict(pool_cov=mean(pool_cov), avg_nwin=mean(nwin), avg_pool=mean(psize))


def main():
    ctx = setup()
    base_m, base_tok = {}, {}
    print("\n===== R0 基线（同 K）=====")
    for K in (4, 6):
        m = agg(baseline(K)); base_m[K], base_tok[K] = m, m["avg_tokens"]
        print(line(f"hybrid@{K}", m))

    results = {}
    for method in ("WS-max", "WS-anchor+max", "WS-mean"):
        ref = "（参照）" if method == "WS-mean" else ""
        print(f"\n===== {method}{ref} =====")
        for K in (4, 6):
            finals, diag = run_method(method, K, ctx)
            m = agg(finals)
            g2 = m["avg_tokens"] <= G2_FACTOR * base_tok[K]
            gate = dev20_pass(finals[DEV.index(DEV20)])
            recall_ok = m["ev_recall"] >= base_m[K]["ev_recall"] - 1e-9
            mrr_ok = m["ev_mrr"] >= base_m[K]["ev_mrr"] - 1e-9
            fx, bk, brk = net_change(finals, baseline(K))
            passed = g2 and gate and recall_ok and mrr_ok and (fx - bk) >= 0
            results[(method, K)] = passed and method != "WS-mean"
            print(f"\nK={K}  (基线 ev-recall={base_m[K]['ev_recall']:.0%}, MRR={base_m[K]['ev_mrr']:.3f}, tok={base_tok[K]:.0f})")
            print(line(f"{method}@{K}", m))
            print(f"    候选窗={diag['avg_nwin']:.1f} 去重池={diag['avg_pool']:.1f} "
                  f"pool-cov={diag['pool_cov']:.0%} | Δrecall={m['ev_recall']-base_m[K]['ev_recall']:+.0%} "
                  f"MRR不劣={'✓' if mrr_ok else '✗'} G2={'✓' if g2 else '✗'} dev20={'✓' if gate else '✗'} "
                  f"净={fx-bk:+d}({brk})")

    print("\n===== 全局验收（WS-max / WS-anchor+max）=====")
    ok = [k for k, v in results.items() if v]
    for (method, K) in [(mm, kk) for mm in ("WS-max", "WS-anchor+max") for kk in (4, 6)]:
        print(f"  ({method}, K={K}) = {'✓ PASS' if results.get((method, K)) else '✗ FAIL'}")
    if ok:
        print(f"\n✓ 通过配置：{ok}")
    else:
        print("\n✗ 无配置同时过 dev20/G2/recall/MRR/净变化 —— 如实记负，不放宽门槛。")


if __name__ == "__main__":
    main()

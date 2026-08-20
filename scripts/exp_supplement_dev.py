"""基线保留式补充检索——第一阶段（无 rerank，完全确定）。dev-only, K_base=6。

base = hybrid top6（冻结）；补充邻居只追加、受逐题 0.5×base token 预算约束。
对照两种确定性候选形态：N-next（idx+1）、N-win（idx-1, idx+1）。

指标用**原始计数**（不用四舍五入百分比判断）；final 上下文可能超过 6 个 chunk，
故指标名为 **augmented final evidence recall**（不叫 recall@6）。
验收：dev20 含 #2；augmented ev-recall 计数 ≥ 22/24（base=21/24）；ev-MRR 不劣；
逐题净变化≥0；平均总 token ≤ 1.5×base。任一不过如实停止。
运行：./.venv/bin/python -m scripts.exp_supplement_dev
"""
import json
from pathlib import Path
from statistics import mean

from app import vectorstore, hybrid, ingest, retrieval_pipeline as rp
from app.eval_metrics import evidence_chunk_rank, reciprocal_rank

DEV = [json.loads(x) for x in Path("eval/dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
K_BASE, P, BUDGET = 6, 12, 0.5
DEV20 = next(it for it in DEV if it["id"] == "dev20")


def toks(chunks):
    return sum(len(c["text"]) // 4 for c in chunks)


def evr(lst, it):
    return evidence_chunk_rank(lst, it["expected_source"], it.get("evidence_quote"))


def dev20_has_2(final):
    return any(c["source"] == "tutorial/path-params.md" and c["chunk_index"] == 2 for c in final)


def setup():
    print("重建索引：reset + 从 docs/ 入库 ...")
    vectorstore.reset(); ingest.ingest_dir("docs")
    print("库内片段:", vectorstore.count())
    return rp.build_lookup()


def eval_finals(finals):
    """返回原始命中计数与指标。"""
    hits = [evr(f, it) is not None for f, it in zip(finals, DEV)]
    mrr = mean(reciprocal_rank(evr(f, it)) for f, it in zip(finals, DEV))
    return sum(hits), mrr, mean(len(f) for f in finals), mean(toks(f) for f in finals)


def main():
    lookup = setup()
    n = len(DEV)

    base_finals = [hybrid.hybrid_search(it["question"], k=K_BASE) for it in DEV]
    b_hits, b_mrr, b_chunks, b_tok = eval_finals(base_finals)
    print(f"\n===== base = hybrid@{K_BASE} =====")
    print(f"  evidence 命中 = {b_hits}/{n}  ev-MRR={b_mrr:.3f}  chunks={b_chunks:.1f}  tok={b_tok:.0f}"
          f"  dev20#2={'✓' if dev20_has_2(base_finals[DEV.index(DEV20)]) else '✗'}")

    any_pass = False
    for mode in ("N-next", "N-win"):
        finals, supp_tok = [], []
        for it in DEV:
            r = rp.run_supplement(it["question"], k_base=K_BASE, mode=mode, p=P,
                                  budget_factor=BUDGET, lookup=lookup)
            finals.append(r["final"]); supp_tok.append(r["supp_tokens"])
        a_hits, a_mrr, a_chunks, a_tok = eval_finals(finals)

        # 逐题净变化（base 冻结，理论上不该有 broken）
        fixed = broken = 0; fx = []; bk = []
        for it, af, bf in zip(DEV, finals, base_finals):
            ah, bh = evr(af, it) is not None, evr(bf, it) is not None
            if ah and not bh:
                fixed += 1; fx.append(it["id"])
            elif bh and not ah:
                broken += 1; bk.append(it["id"])

        gate = dev20_has_2(finals[DEV.index(DEV20)])
        recall_ok = a_hits >= b_hits + 1                 # 原始计数 ≥ 22/24
        mrr_ok = a_mrr >= b_mrr - 1e-9
        g2 = a_tok <= 1.5 * b_tok
        net_ok = (fixed - broken) >= 0
        passed = gate and recall_ok and mrr_ok and g2 and net_ok
        any_pass = any_pass or passed

        print(f"\n===== {mode} =====")
        print(f"  augmented ev 命中 = {a_hits}/{n} (base {b_hits}/{n})  ev-MRR={a_mrr:.3f}  "
              f"chunks={a_chunks:.1f}  tok={a_tok:.0f}  补充tok(均)={mean(supp_tok):.0f}")
        print(f"  修好{fixed}{fx}  弄坏{broken}{bk}  净={fixed-broken:+d}")
        print(f"  门槛: dev20#2={'✓' if gate else '✗'}  命中≥{b_hits+1}={'✓' if recall_ok else '✗'}  "
              f"MRR不劣={'✓' if mrr_ok else '✗'}  G2(≤{1.5*b_tok:.0f})={'✓' if g2 else '✗'}  "
              f"净≥0={'✓' if net_ok else '✗'}  → {'✓ PASS' if passed else '✗ FAIL'}")

    print("\n===== 结论 =====")
    if any_pass:
        print("✓ 存在确定性无-rerank 配置通过全部门槛 → 第一阶段成功，不引入模型。")
    else:
        print("✗ N-next / N-win 均未通过 → 第一阶段失败。按计划下一步为 DeepSeek rerank + N-win（另行确认）。")


if __name__ == "__main__":
    main()

"""第二阶段：DeepSeek rerank + N-win（dev-only, K_base=6）。需要 DeepSeek key。

流程：base=hybrid@6（冻结）→ N-win 候选 → DeepSeek rerank 只排补充单元 →
按 rerank 顺序在逐题 0.5×base token 预算内原子加入。解析失败回退确定性 N-win 序。

验收（原始计数）：dev20 含 path-params.md#2；augmented evidence 命中 ≥ 22/24；
ev-MRR ≥ base；无 broken；总 token ≤ 1.5×base。任一失败如实停止。

**默认 cache-only**：不加 --allow-api 时，缓存缺失/损坏一律 api_disabled 回退，
绝不调用真实 DeepSeek。只有显式 `--allow-api` 才允许真实调用：
运行：./.venv/bin/python -m scripts.exp_rerank_dev            # cache-only（默认）
      ./.venv/bin/python -m scripts.exp_rerank_dev --allow-api  # 允许真实调用
"""
import argparse
import json
from pathlib import Path
from statistics import mean

from app import vectorstore, hybrid, ingest, config, rerank, retrieval_pipeline as rp
from app.eval_metrics import evidence_chunk_rank, source_rank, reciprocal_rank

DEV = [json.loads(x) for x in Path("eval/dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
K_BASE, P, BUDGET, MODE = 6, 12, 0.5, "N-win"
DEV20 = next(it for it in DEV if it["id"] == "dev20")


def toks(chunks):
    return sum(len(c["text"]) // 4 for c in chunks)


def evr(lst, it):
    return evidence_chunk_rank(lst, it["expected_source"], it.get("evidence_quote"))


def dev20_has_2(final):
    return any(c["source"] == "tutorial/path-params.md" and c["chunk_index"] == 2 for c in final)


def completeness_checks(n, cached, parse_fail, cache_err, api_err, api_disabled, this_run_calls):
    """完整性条件（纯函数，便于单测）：全部为 True 才允许报告 PASS。"""
    return {
        "24/24 有效缓存": cached == n,
        "parse_fail=0": parse_fail == 0,
        "cache_error=0": cache_err == 0,
        "api_error=0": api_err == 0,
        "api_disabled=0": api_disabled == 0,
        "本次api_calls=0": this_run_calls == 0,
    }


def _result_code(checks: dict, passed) -> int:
    """退出码（纯函数）：不完整→2；完整但指标 FAIL→1；完整且 PASS→0。"""
    if not all(checks.values()):
        return 2
    return 0 if passed else 1


def eval_finals(finals):
    hits = sum(evr(f, it) is not None for f, it in zip(finals, DEV))
    mrr = mean(reciprocal_rank(evr(f, it)) for f, it in zip(finals, DEV))
    src_hits = sum(source_rank(f, it["expected_source"]) is not None for f, it in zip(finals, DEV))
    return hits, mrr, src_hits, mean(len(f) for f in finals), mean(toks(f) for f in finals)


def _key_ok() -> bool:
    k = config.DEEPSEEK_API_KEY
    if not k or k.strip() == "sk-你的key":
        print("✗ DEEPSEEK_API_KEY 未配置或仍是占位符（.env 里的 sk-你的key）。请填入真实 key。")
        return False
    if not k.isascii():
        print("✗ DEEPSEEK_API_KEY 含非 ASCII 字符，疑似占位符/损坏。请检查 .env。")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-api", action="store_true",
                    help="允许真实 DeepSeek 调用（默认 cache-only，缺失/坏缓存一律回退，不调用）")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="reset Chroma 并从 docs/ 重新入库（默认复用现有索引，不修改）")
    args = ap.parse_args()

    if args.allow_api:
        if not _key_ok():
            return 2                     # --allow-api 但 key 无效 → 非零
        print("⚠ --allow-api 已开启：缓存缺失/损坏时将发起真实 DeepSeek 调用。")
    else:
        print("cache-only 模式（默认）：不调用真实 DeepSeek；缓存缺失/损坏 → api_disabled 回退。")

    # 默认复用现有索引，不 reset；仅 --rebuild-index 或空库时才入库
    if args.rebuild_index:
        print("索引：--rebuild-index → reset + 从 docs/ 入库（已修改本地索引）")
        vectorstore.reset(); ingest.ingest_dir("docs")
    elif vectorstore.count() == 0:
        print("索引：当前为空 → 从 docs/ 入库（新建，未 reset 已有数据）")
        ingest.ingest_dir("docs")
    else:
        print("索引：复用现有索引（未修改）")
    print("库内片段:", vectorstore.count())
    lookup = rp.build_lookup(); n = len(DEV)

    base_finals = [hybrid.hybrid_search(it["question"], k=K_BASE) for it in DEV]
    b_hits, b_mrr, b_src, b_chunks, b_tok = eval_finals(base_finals)
    print(f"\n===== base = hybrid@{K_BASE} =====")
    print(f"  evidence 命中={b_hits}/{n}  ev-MRR={b_mrr:.3f}  source 命中={b_src}/{n}  "
          f"chunks={b_chunks:.1f}  tok={b_tok:.0f}")

    finals = []
    this_run_calls = 0                 # 本次真正发起的网络调用（未命中缓存时才可能 >0）
    hist_calls = 0                     # 缓存记录里的历史调用（与本次分开）
    hist_in = hist_out = 0
    parse_fail = api_err = cache_err = api_disabled = cached = 0
    for it in DEV:
        base = hybrid.hybrid_search(it["question"], k=K_BASE)
        anchors = hybrid.hybrid_search(it["question"], k=P, top_n=20)
        cands = rp.build_candidates(base, anchors, lookup, MODE)
        rr = rerank.rerank_units(it["question"], cands, use_cache=True,
                                 allow_api=args.allow_api)   # 默认 cache-only
        t = rr["telemetry"]
        if rr["cached"]:
            cached += 1
            hist_calls += t["api_calls"]; hist_in += t["in_tokens"]; hist_out += t["out_tokens"]
        else:
            this_run_calls += t["api_calls"]   # cache-only 下应恒为 0（api_disabled）
        if not rr["parse_ok"]:
            parse_fail += 1
        if (rr["error"] or "").startswith("api_error"):
            api_err += 1
        if rr["error"] == "api_disabled":
            api_disabled += 1
        if rr.get("cache_error"):
            cache_err += 1
        order = rr["order"]                       # None → supplement_base 回退确定性序
        r = rp.supplement_base(base, anchors, lookup, MODE, BUDGET, order_unit_ids=order)
        finals.append(r["final"])

    # ── 完整性闸门：仅当以下全部满足才允许报告 PASS，否则标记不完整并停止 ──
    #   24/24 有效缓存排序、parse_fail=0、cache_error=0、api_error=0、
    #   api_disabled=0、本次真实调用=0
    checks = completeness_checks(n, cached, parse_fail, cache_err, api_err,
                                 api_disabled, this_run_calls)
    incomplete = not all(checks.values())
    print(f"\n===== 调用核算（本次 vs 历史）=====")
    print(f"  本次真实网络调用 = {this_run_calls}")
    print(f"  缓存命中 = {cached}/{n}   API禁用(回退) = {api_disabled}")
    print(f"  历史缓存调用(不计入本次) = {hist_calls}  hist_in_tok={hist_in}  hist_out_tok={hist_out}")
    print(f"  容错: 解析失败={parse_fail}  API异常={api_err}  坏缓存={cache_err}")
    print("  完整性检查: " + "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in checks.items()))

    if incomplete:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n===== 结论：不完整实验 =====")
        print(f"✗ 未满足完整性条件: {failed} —— 标记为不完整实验并停止，不补调 DeepSeek。")
        return _result_code(checks, None)   # 不完整 → 2

    # 缓存完整（24/24）时才计算并报告指标
    a_hits, a_mrr, a_src, a_chunks, a_tok = eval_finals(finals)
    fixed = broken = 0; fx = []; bk = []
    for it, af, bf in zip(DEV, finals, base_finals):
        ah, bh = evr(af, it) is not None, evr(bf, it) is not None
        if ah and not bh:
            fixed += 1; fx.append(it["id"])
        elif bh and not ah:
            broken += 1; bk.append(it["id"])

    gate = dev20_has_2(finals[DEV.index(DEV20)])
    recall_ok = a_hits >= b_hits + 1
    mrr_ok = a_mrr >= b_mrr - 1e-9
    g2 = a_tok <= 1.5 * b_tok
    net_ok = broken == 0
    passed = gate and recall_ok and mrr_ok and g2 and net_ok

    print(f"\n===== rerank + {MODE}（cache-only）=====")
    print(f"  augmented evidence 命中={a_hits}/{n} (base {b_hits}/{n})  ev-MRR={a_mrr:.3f}  "
          f"source 命中={a_src}/{n}  chunks={a_chunks:.1f}  tok={a_tok:.0f}")
    print(f"  逐题：修好{fixed}{fx}  弄坏{broken}{bk}")
    print(f"\n  门槛: dev20#2={'✓' if gate else '✗'}  命中≥{b_hits+1}={'✓' if recall_ok else '✗'}  "
          f"MRR不劣={'✓' if mrr_ok else '✗'}  G2(≤{1.5*b_tok:.0f})={'✓' if g2 else '✗'}  "
          f"无broken={'✓' if net_ok else '✗'}")
    print("\n===== 结论 =====")
    print("✓ PASS：rerank + N-win 通过全部门槛。" if passed
          else "✗ FAIL：未通过全部门槛 —— 如实停止，不放宽标准。")
    return _result_code(checks, passed)   # 完整且 PASS → 0；指标 FAIL → 1


if __name__ == "__main__":
    raise SystemExit(main())

"""检索评测：纯向量 vs 混合检索。

同时报告两套独立指标（见 app.eval_metrics）：
- source：按 source 去重后的 recall@1/3 与 MRR
- evidence：证据片段级（不去重）的 recall@1/3 与 MRR

默认只跑 dev（可据此调参）；test 为冻结集，仅里程碑评估。
里程碑评估请加 --rebuild-index 以干净索引运行，避免旧库污染数字：
    ./.venv/bin/python -m scripts.evaluate                          # dev
    ./.venv/bin/python -m scripts.evaluate --by-type                # 按类型
    ./.venv/bin/python -m scripts.evaluate --split test --rebuild-index  # 里程碑
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from app import vectorstore, hybrid, ingest
from app.eval_metrics import source_rank, evidence_chunk_rank, aggregate

EVAL_DIR = Path("eval")
DOCS_DIR = "docs"
TOP_N = 10


def load(split: str) -> list[dict]:
    path = EVAL_DIR / f"{split}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


RETRIEVERS = {
    "vector": lambda q: vectorstore.search(q, k=TOP_N),
    "hybrid": lambda q: hybrid.hybrid_search(q, k=TOP_N),
}


def _metrics(items, retrieve):
    src, evi = [], []
    for it in items:
        ranked = retrieve(it["question"])
        src.append(source_rank(ranked, it["expected_source"]))
        evi.append(evidence_chunk_rank(ranked, it["expected_source"], it.get("evidence_quote")))
    return aggregate(src), aggregate(evi)


def _ensure_index(rebuild: bool):
    if rebuild:
        print("重建索引：reset Chroma 并从 docs/ 重新入库 ...")
        vectorstore.reset()
        ingest.ingest_dir(DOCS_DIR)
    elif vectorstore.count() == 0:
        ingest.ingest_dir(DOCS_DIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="reset Chroma 并从 docs/ 重新入库后再评测（里程碑必须用）")
    ap.add_argument("--by-type", action="store_true", help="按问题类型拆分 evidence recall@3")
    args = ap.parse_args()

    _ensure_index(args.rebuild_index)
    items = load(args.split)
    if args.split == "test":
        print("⚠ 正在评估冻结的 test 集——仅用于里程碑报告，不要据此调参。")
    print(f"\n[{args.split}] {len(items)} 条 | top_n={TOP_N} | 库内片段 {vectorstore.count()}\n")
    hdr = f"{'检索':<8}| {'src@1':>6} {'src@3':>6} {'srcMRR':>7} | {'ev@1':>6} {'ev@3':>6} {'evMRR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, fn in RETRIEVERS.items():
        s, e = _metrics(items, fn)
        print(f"{name:<8}| {s['recall@1']:>6.0%} {s['recall@3']:>6.0%} {s['mrr']:>7.3f} | "
              f"{e['recall@1']:>6.0%} {e['recall@3']:>6.0%} {e['mrr']:>7.3f}")

    if args.by_type:
        print("\n按问题类型（hybrid, evidence recall@3）:")
        buckets = defaultdict(list)
        for it in items:
            buckets[it.get("type", "?")].append(it)
        for t, group in sorted(buckets.items()):
            _, e = _metrics(group, RETRIEVERS["hybrid"])
            print(f"  {t:<10} n={len(group):<3} ev-recall@3={e['recall@3']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

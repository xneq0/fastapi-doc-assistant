"""对比『纯向量检索』 vs 『混合检索』的命中率(recall@k)。
运行: ./.venv/bin/python -m scripts.evaluate
"""
from app import vectorstore, hybrid, ingest
from app.eval_set import EVAL

K = 1


def hit(sources: list[str], expected: str) -> bool:
    return expected in sources


def run():
    if vectorstore.count() == 0:
        print("库为空，先导入 docs/ ...")
        ingest.ingest_dir("docs")

    vec_hits = hybrid_hits = 0
    print(f"评测集共 {len(EVAL)} 条，指标 = recall@{K}（正确文档是否出现在 top-{K}）\n")
    print(f"{'问题':<52}{'向量':<6}{'混合'}")
    print("-" * 68)
    for item in EVAL:
        q, exp = item["q"], item["src"]
        vec_src = [h["source"] for h in vectorstore.search(q, k=K)]
        hyb_src = [h["source"] for h in hybrid.hybrid_search(q, k=K)]
        v, h = hit(vec_src, exp), hit(hyb_src, exp)
        vec_hits += v
        hybrid_hits += h
        print(f"{q[:50]:<52}{'✓' if v else '✗':<6}{'✓' if h else '✗'}")

    n = len(EVAL)
    print("-" * 68)
    print(f"\n纯向量  recall@{K} = {vec_hits}/{n} = {vec_hits/n:.0%}")
    print(f"混合    recall@{K} = {hybrid_hits}/{n} = {hybrid_hits/n:.0%}")
    delta = (hybrid_hits - vec_hits) / n
    print(f"提升    = {delta:+.0%}")


if __name__ == "__main__":
    run()

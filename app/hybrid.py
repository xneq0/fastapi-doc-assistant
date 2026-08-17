"""混合检索：BM25 关键词召回 + 向量语义召回，用 RRF 融合。

为什么要混合：
- 纯向量：擅长语义近似，但对精确关键词（函数名、报错、API 名）不敏感。
- 纯 BM25：擅长精确关键词，但不懂同义/语义。
- RRF（Reciprocal Rank Fusion）：把两路排名融合，取长补短。
"""
import re
from rank_bm25 import BM25Okapi
from . import vectorstore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _bm25_rank(query: str, corpus: list[dict], top_n: int) -> list[str]:
    q_tokens = set(_tokenize(query))
    tokenized = [_tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    # 用「query 词与文档词是否有交集」判断是否命中，而不是 score>0：
    # BM25Okapi 的 IDF 在某些语料（如词恰好出现在一半文档）会退化为 0，
    # 导致精确匹配也被 score>0 误过滤。有 token 交集才算真正含关键词。
    candidates = [
        (c, s) for c, toks, s in zip(corpus, tokenized, scores)
        if q_tokens & set(toks)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c["id"] for c, _ in candidates[:top_n]]


def hybrid_search(query: str, k: int = 4, top_n: int = 10, rrf_k: int = 60) -> list[dict]:
    corpus = vectorstore.get_all()
    by_id = {c["id"]: c for c in corpus}

    # 两路召回
    vec_ids = [h["id"] for h in vectorstore.search(query, k=top_n)]
    bm25_ids = _bm25_rank(query, corpus, top_n)

    # RRF 融合：score = Σ 1 / (rrf_k + rank)
    fused: dict[str, float] = {}
    for rank, cid in enumerate(vec_ids):
        fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)
    for rank, cid in enumerate(bm25_ids):
        fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)

    top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    out = []
    for cid in top_ids:
        c = by_id[cid]
        out.append({"id": cid, "text": c["text"], "source": c["source"],
                    "chunk_index": c["chunk_index"], "score": round(fused[cid], 5)})
    return out

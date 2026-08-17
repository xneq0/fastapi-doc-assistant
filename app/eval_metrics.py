"""评测指标核心（纯函数，无检索/网络依赖，便于单测）。

刻意拆成两套**互相独立**的名次指标，避免一个函数同时承担两种语义：

- source_rank：只按 source 首次出现去重后的名次；**不看证据**。
  衡量「正确文档整体排到第几」。
- evidence_chunk_rank：**不按 source 去重**，在原始 chunk 排名里，
  第一个「来源正确且包含 evidence_quote」的 chunk 的名次。
  衡量「真正能回答问题的证据片段排到第几」。

二者可能不同：例如 [A(无证据), B, A(有证据)] 中
source_rank(A)=1，但 evidence_chunk_rank(A)=3。
"""
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def source_rank(ranked: list[dict], expected_source: str) -> int | None:
    """按 source 首次出现去重后，expected_source 的 1-based 名次；不在结果里返回 None。"""
    seen: list[str] = []
    for c in ranked:
        if c["source"] not in seen:
            seen.append(c["source"])
    return seen.index(expected_source) + 1 if expected_source in seen else None


def evidence_chunk_rank(ranked: list[dict], expected_source: str,
                        evidence_quote: str | None) -> int | None:
    """原始 chunk 排名中，第一个「来源正确且含证据」的 chunk 的 1-based 名次；无则 None。

    不按 source 去重：只命中文件、但片段里没有证据的 chunk 不计。
    """
    ev = _norm(evidence_quote) if evidence_quote else None
    for i, c in enumerate(ranked):
        if c["source"] == expected_source and (ev is None or ev in _norm(c["text"])):
            return i + 1
    return None


def recall_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def aggregate(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {
        "recall@1": sum(recall_at(r, 1) for r in ranks) / n,
        "recall@3": sum(recall_at(r, 3) for r in ranks) / n,
        "mrr": sum(reciprocal_rank(r) for r in ranks) / n,
    }

"""评测指标纯函数单测（离线，无检索/LLM）。"""
from app.eval_metrics import (
    source_rank, evidence_chunk_rank, recall_at, reciprocal_rank, aggregate,
)


def _chunks(*pairs):
    return [{"source": s, "text": t} for s, t in pairs]


def test_source_and_evidence_ranks_differ():
    # Codex 指定回归：A(无证据), B, A(有证据)
    ranked = _chunks(
        ("a.md", "intro paragraph, no evidence here"),
        ("b.md", "some other doc"),
        ("a.md", "here is the needle"),
    )
    # source 去重后 a 首次出现在第 1 位
    assert source_rank(ranked, "a.md") == 1
    # 证据片段级：真正含证据的 a 片段在第 3 位
    assert evidence_chunk_rank(ranked, "a.md", "the needle") == 3
    # 二者不能都算成 1
    assert source_rank(ranked, "a.md") != evidence_chunk_rank(ranked, "a.md", "the needle")


def test_source_rank_dedup_and_miss():
    ranked = _chunks(("a.md", "x"), ("a.md", "y"), ("b.md", "z"))
    assert source_rank(ranked, "b.md") == 2      # a 去重占 1 名，b 第 2
    assert source_rank(ranked, "c.md") is None   # 不在结果里


def test_evidence_rank_requires_correct_source_and_quote():
    ranked = _chunks(("b.md", "the needle is here"), ("a.md", "no evidence"))
    assert evidence_chunk_rank(ranked, "a.md", "the needle") is None  # 证据在错文件不算
    ranked = _chunks(("a.md", "contains the needle"))
    assert evidence_chunk_rank(ranked, "a.md", "the needle") == 1


def test_evidence_rank_no_quote_falls_back_to_source():
    ranked = _chunks(("b.md", "x"), ("a.md", "y"))
    assert evidence_chunk_rank(ranked, "a.md", None) == 2


def test_recall_and_rr():
    assert recall_at(1, 1) and recall_at(3, 3) and not recall_at(4, 3)
    assert not recall_at(None, 3)
    assert reciprocal_rank(2) == 0.5 and reciprocal_rank(None) == 0.0


def test_aggregate():
    m = aggregate([1, 2, None, 3])
    assert m["recall@1"] == 0.25
    assert m["recall@3"] == 0.75
    assert abs(m["mrr"] - (1 + 0.5 + 0 + 1 / 3) / 4) < 1e-9

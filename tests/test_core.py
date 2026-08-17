"""核心逻辑的自动化测试。

用假 embedding + 临时向量库，不依赖 Ollama / DeepSeek，任何环境可跑：
    ./.venv/bin/python -m pytest -q
"""
import os
import hashlib
import tempfile

# 必须在导入 app 之前把向量库指向临时目录，避免污染真实数据
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()

from app import embeddings, ingest, vectorstore, hybrid  # noqa: E402
from app.ingest import chunk_text  # noqa: E402
from app.hybrid import _bm25_rank, _tokenize  # noqa: E402


def _fake_embed(texts):
    """确定性假向量（16 维），让测试无需真实 embedding 服务。"""
    return [[b / 255.0 for b in hashlib.md5(t.encode()).digest()] for t in texts]


embeddings.embed_texts = _fake_embed
embeddings.embed_query = lambda t: _fake_embed([t])[0]


def test_chunk_text_overlap():
    chunks = chunk_text("abcdefghij" * 30, size=100, overlap=20)
    assert len(chunks) >= 3
    assert all(chunks)


def test_tokenize():
    assert _tokenize("HTTPException 422!") == ["httpexception", "422"]


def test_bm25_excludes_non_matching():
    corpus = [
        {"id": "a", "text": "background tasks add_task notification"},
        {"id": "b", "text": "query parameters skip limit"},
        {"id": "c", "text": "dependency injection depends"},
    ]
    # 只有 a 与 query 有 token 交集，b/c 应被过滤
    assert _bm25_rank("background add_task", corpus, top_n=10) == ["a"]


def test_bm25_keeps_exact_match_when_idf_zero():
    # 回归：两篇文档、query 恰好只匹配一篇时，BM25Okapi 的 IDF 会退化为 0，
    # 精确匹配的 foo 分数为 0。按 token 交集过滤应仍保留 foo，而非返回空。
    corpus = [
        {"id": "foo", "text": "foo foo foo"},
        {"id": "bar", "text": "bar bar bar"},
    ]
    assert _bm25_rank("foo", corpus, top_n=10) == ["foo"]


def test_ingest_is_idempotent(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("HTTPException raises a 404 error not found")
    (d / "b.md").write_text("query parameter skip limit default value")

    vectorstore.reset()
    ingest.ingest_dir(str(d))
    first = vectorstore.count()
    ingest.ingest_dir(str(d))  # 重复导入
    second = vectorstore.count()

    assert first > 0
    assert first == second  # 幂等：重复入库数量不增长


def test_source_uses_relative_path(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.md").write_text("nested doc about HTTPException")

    vectorstore.reset()
    ingest.ingest_dir(str(d))
    sources = {c["source"] for c in vectorstore.get_all()}
    assert "sub/a.md" in sources  # 保留子目录，避免同名冲突


def test_hybrid_search_smoke(tmp_path):
    # 用 4 篇文档，让关键词只出现在少数文档里，避免 BM25 IDF 退化为 0
    d = tmp_path / "docs"
    d.mkdir()
    (d / "err.md").write_text("Use HTTPException to return a 404 error")
    (d / "q.md").write_text("skip and limit are query parameters")
    (d / "body.md").write_text("send JSON request body with a Pydantic model")
    (d / "bg.md").write_text("run background tasks after the response")

    vectorstore.reset()
    ingest.ingest_dir(str(d))
    hits = hybrid.hybrid_search("HTTPException", k=2)
    assert hits and "source" in hits[0]
    # BM25 精确命中唯一含该关键词的文档，应排第一（不受假向量噪声影响）
    assert hits[0]["source"] == "err.md"

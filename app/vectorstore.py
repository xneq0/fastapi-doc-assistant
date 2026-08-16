"""Chroma 向量库封装。embedding 由我们自己算好后传入（不依赖 Chroma 内置模型）。"""
import chromadb
from . import config, embeddings

_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(chunks: list[dict]):
    """chunks: [{id, text, source, chunk_index}]

    用 upsert 而非 add：相同 id 会覆盖而不是报 DuplicateIDError，
    因此重复调用 /ingest 是幂等的。
    """
    vectors = embeddings.embed_texts([c["text"] for c in chunks])
    _collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )


def search(query: str, k: int = 4) -> list[dict]:
    qvec = embeddings.embed_query(query)
    res = _collection.query(query_embeddings=[qvec], n_results=k)
    out = []
    for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0]):
        out.append({"id": cid, "text": text, "source": meta["source"],
                    "chunk_index": meta["chunk_index"], "distance": dist})
    return out


def get_all() -> list[dict]:
    """取出库内全部片段，供 BM25 关键词检索建索引用。"""
    res = _collection.get(include=["documents", "metadatas"])
    return [
        {"id": cid, "text": text, "source": meta["source"], "chunk_index": meta["chunk_index"]}
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def count() -> int:
    return _collection.count()


def reset():
    global _collection
    _client.delete_collection(config.COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

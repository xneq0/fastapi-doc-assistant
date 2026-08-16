"""本地 embedding：调用 Ollama，避免额外的付费 API 和 torch 依赖。"""
import httpx
from . import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量。使用 Ollama 的 /api/embed 批量接口。"""
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]

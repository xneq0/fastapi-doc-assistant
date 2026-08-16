"""RAG 核心：检索 -> 拼上下文 -> 让 DeepSeek 基于文档作答并标注出处。"""
from . import vectorstore, hybrid, llm

SYSTEM = (
    "你是一个开发者文档问答助手。只依据给定的【文档片段】回答问题，"
    "不要编造。若文档中找不到答案，明确说『文档中未提及』。"
    "回答末尾用 [来源: 文件名#片段号] 的形式标注你引用了哪些片段。"
)


def answer(question: str, k: int = 4, use_hybrid: bool = True) -> dict:
    hits = hybrid.hybrid_search(question, k=k) if use_hybrid else vectorstore.search(question, k=k)
    if not hits:
        return {"answer": "知识库为空，请先导入文档。", "sources": []}

    context = "\n\n".join(
        f"[{h['source']}#{h['chunk_index']}]\n{h['text']}" for h in hits
    )
    user = f"【文档片段】\n{context}\n\n【问题】\n{question}"
    reply = llm.chat(SYSTEM, user)
    return {
        "answer": reply,
        "retrieval": "hybrid" if use_hybrid else "vector",
        "sources": [
            {"source": h["source"], "chunk_index": h["chunk_index"]}
            for h in hits
        ],
    }

"""端到端测试检索链路（不需要 DeepSeek key，只测 embedding + 向量库 + 检索）。
运行: ./.venv/bin/python -m scripts.test_retrieval
"""
from app import ingest, vectorstore

print("1) 重置并导入 docs/ ...")
vectorstore.reset()
result = ingest.ingest_dir("docs")
print("   导入结果:", result, "| 总片段数:", vectorstore.count())

print("\n2) 检索测试:")
for q in ["How do I make a query parameter required?",
          "What is dependency injection used for?"]:
    print(f"\n   Q: {q}")
    for h in vectorstore.search(q, k=2):
        preview = h["text"][:70].replace("\n", " ")
        print(f"     -> [{h['source']}#{h['chunk_index']}] dist={h['distance']:.4f} | {preview}...")

"""检索链路手动演示（不需要 DeepSeek key，只跑 embedding + 向量库 + 检索）。
运行: ./.venv/bin/python -m scripts.demo_retrieval

注意：这是手动演示脚本，不是 pytest 用例（故不以 test_ 命名，
且执行代码放在 main() 里，import 时不产生副作用）。
"""
from app import ingest, vectorstore


def main():
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


if __name__ == "__main__":
    main()

"""不可答问题的拒答评测（需要 DeepSeek API key，默认手动运行）。

对 eval/unanswerable.jsonl 里「知识库中没有答案」的问题实际调用问答链路，
把每条的原始回答保存到 eval/outputs/ 供人工复核。启发式只作参考、不作为
最终结论——是否真的正确拒答需人工看原文判断。

    ./.venv/bin/python -m scripts.evaluate_answers
"""
import json
import re
from datetime import datetime
from pathlib import Path

from app import rag, vectorstore, ingest, config

EVAL_FILE = Path("eval/unanswerable.jsonl")
OUT_DIR = Path("eval/outputs")

# 仅作参考的拒答启发式；真正判定以人工复核 saved 文件为准
REFUSAL_HINTS = ["文档中未提及", "未提及", "没有相关", "无法在文档", "not mentioned",
                 "not in the", "no relevant", "cannot find", "couldn't find"]


def looks_abstained(answer: str) -> bool:
    low = answer.lower()
    return any(h.lower() in low for h in REFUSAL_HINTS)


def main() -> int:
    if not config.DEEPSEEK_API_KEY:
        print("未配置 DEEPSEEK_API_KEY，跳过。请先在 .env 填入 key。")
        return 1
    if vectorstore.count() == 0:
        ingest.ingest_dir("docs")

    items = [json.loads(x) for x in EVAL_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"unanswerable_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    abstained = 0
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            res = rag.answer(it["question"])
            hint = looks_abstained(res["answer"])
            abstained += hint
            record = {
                "id": it["id"], "type": it["type"], "question": it["question"],
                "answer": res["answer"], "sources": res["sources"],
                "heuristic_abstained": hint,   # 仅参考，需人工确认
                "human_verdict": None,          # 复核时填 true/false
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {it['id']} [{'拒答?' if hint else '疑似作答!'}] {it['question'][:50]}")

    n = len(items)
    print(f"\n启发式拒答率: {abstained}/{n} = {abstained/n:.0%}（仅参考）")
    print(f"原始回答已保存: {out_path}")
    print("请人工打开该文件复核每条 answer，并在 human_verdict 标注是否真的正确拒答。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

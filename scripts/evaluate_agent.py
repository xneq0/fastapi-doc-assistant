"""Agent dev-only 基线评测；会调用真实 DeepSeek API，不读取冻结 test 集。"""
import argparse
import json
from pathlib import Path

from app import agent, config, vectorstore


DATA_PATH = Path("eval/agent_dev.jsonl")
OUTPUT_PATH = Path("eval/outputs/agent_dev_baseline.json")
VARIANTS = {
    "baseline": False,
    "english-query": True,
}


def tool_selection_metrics(expected_tools: list[str], trace: list[dict]) -> dict:
    """逐题工具指标：必需工具是否全调用、是否出现期望外工具。"""
    expected = set(expected_tools)
    actual_order = [row.get("tool") for row in trace if isinstance(row, dict) and row.get("tool")]
    actual = set(actual_order)
    return {
        "actual_tools": actual_order,
        "required_tools_recalled": expected.issubset(actual),
        "has_extra_tool_call": bool(actual - expected),
    }


def source_metrics(expected_source: str | None, search_results: list[list[dict]]) -> dict:
    """按 search_docs 调用顺序计算 expected_source 是否命中及首个结果是否 top-1。"""
    if expected_source is None:
        return {"source_hit": None, "source_top1": None}

    valid_calls = [items for items in search_results if isinstance(items, list)]
    sources = [
        item.get("source")
        for items in valid_calls
        for item in items
        if isinstance(item, dict)
    ]
    first_source = next(
        (
            items[0].get("source")
            for items in valid_calls
            if items and isinstance(items[0], dict)
        ),
        None,
    )
    return {
        "source_hit": expected_source in sources,
        "source_top1": first_source == expected_source,
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    """汇总总体、分类工具指标，以及概念题 source 指标。"""
    def summarize(group: list[dict]) -> dict:
        n = len(group)
        return {
            "n": n,
            "required_tool_recall": (
                sum(row["required_tools_recalled"] for row in group) / n if n else 0.0
            ),
            "extra_call_rate": (
                sum(row["has_extra_tool_call"] for row in group) / n if n else 0.0
            ),
            "avg_steps": sum(row["steps"] for row in group) / n if n else 0.0,
        }

    categories = sorted({row["category"] for row in rows})
    concept = [row for row in rows if row["category"] == "concept"]
    source_n = len(concept)
    return {
        "overall": summarize(rows),
        "by_category": {
            category: summarize([row for row in rows if row["category"] == category])
            for category in categories
        },
        "concept_source": {
            "n": source_n,
            "hit_rate": (
                sum(row["source_hit"] is True for row in concept) / source_n
                if source_n else 0.0
            ),
            "top1_rate": (
                sum(row["source_top1"] is True for row in concept) / source_n
                if source_n else 0.0
            ),
        },
    }


def _load_dataset() -> list[dict]:
    return [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_with_search_capture(question: str, *, english_search_query: bool = False):
    captured = []
    original = agent._TOOL_FUNCTIONS["search_docs"]

    def capture_search_docs(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    agent._TOOL_FUNCTIONS["search_docs"] = capture_search_docs
    try:
        result = agent.run_agent(
            question,
            english_search_query=english_search_query,
        )
    finally:
        agent._TOOL_FUNCTIONS["search_docs"] = original
    return result, captured


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="baseline",
        help="baseline 保持原工具说明；english-query 要求使用英文检索词。",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        help="重复实验编号；提供后按变体和编号生成独立输出文件。",
    )
    return parser.parse_args(argv)


def _output_path(variant: str, run_index: int | None) -> Path:
    if run_index is None:
        return OUTPUT_PATH if variant == "baseline" else Path(
            "eval/outputs/agent_dev_english_query.json"
        )
    return Path(f"eval/outputs/agent_dev_{variant.replace('-', '_')}_run{run_index}.json")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_index is not None and args.run_index < 1:
        print("✗ --run-index 必须是正整数。")
        return 2

    key = config.DEEPSEEK_API_KEY.strip()
    if not key or key == "sk-你的key":
        print("✗ 缺少有效的 DEEPSEEK_API_KEY。")
        return 2
    if vectorstore.count() == 0:
        print("✗ 当前向量库为空，请先导入 docs/。")
        return 2

    dataset = _load_dataset()
    english_search_query = VARIANTS[args.variant]
    rows = []
    for index, item in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {item['id']}")
        try:
            result, search_results = _run_with_search_capture(
                item["question"],
                english_search_query=english_search_query,
            )
            error_type = None
        except Exception as exc:  # noqa: BLE001 只记录稳定类型，不输出异常详情
            result = {"answer": "", "trace": [], "steps": 0}
            search_results = []
            error_type = type(exc).__name__

        row = {
            **item,
            "answer": result["answer"],
            "trace": result["trace"],
            "steps": result["steps"],
            "error_type": error_type,
            "search_results": search_results,
        }
        row.update(tool_selection_metrics(item["expected_tools"], result["trace"]))
        row.update(source_metrics(item.get("expected_source"), search_results))
        rows.append(row)

    summary = aggregate_metrics(rows)
    summary["n_errors"] = sum(row["error_type"] is not None for row in rows)
    output = {
        "dataset": str(DATA_PATH),
        "model": config.DEEPSEEK_CHAT_MODEL,
        "variant": args.variant,
        "run_index": args.run_index,
        "summary": summary,
        "results": rows,
    }
    output_path = _output_path(args.variant, args.run_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✓ 原始回答与 trace 已保存到 {output_path}")
    return 0 if summary["n_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

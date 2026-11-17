"""Agent 评测指标纯函数测试；不调用模型或网络。"""
import os
import tempfile

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["DEEPSEEK_API_KEY"] = "test-key"

from scripts.evaluate_agent import (  # noqa: E402
    aggregate_metrics,
    source_metrics,
    tool_selection_metrics,
)


def test_tool_selection_metrics_separates_recall_and_extra_calls():
    trace = [{"tool": "search_docs"}, {"tool": "lookup_api"}, {"tool": "search_docs"}]

    exact = tool_selection_metrics(["search_docs", "lookup_api"], trace)
    assert exact["required_tools_recalled"] is True
    assert exact["has_extra_tool_call"] is False

    missing = tool_selection_metrics(["search_docs", "lookup_api"], [{"tool": "search_docs"}])
    assert missing["required_tools_recalled"] is False
    assert missing["has_extra_tool_call"] is False

    extra = tool_selection_metrics(["search_docs"], trace)
    assert extra["required_tools_recalled"] is True
    assert extra["has_extra_tool_call"] is True


def test_source_metrics_uses_raw_results_and_first_returned_source():
    calls = [
        [
            {"source": "tutorial/query-params.md", "chunk_id": "a", "text": "x"},
            {"source": "tutorial/body.md", "chunk_id": "b", "text": "y"},
        ]
    ]

    assert source_metrics("tutorial/query-params.md", calls) == {
        "source_hit": True,
        "source_top1": True,
    }
    assert source_metrics("tutorial/body.md", calls) == {
        "source_hit": True,
        "source_top1": False,
    }
    assert source_metrics("tutorial/missing.md", calls) == {
        "source_hit": False,
        "source_top1": False,
    }


def test_aggregate_metrics_reports_overall_category_and_concept_source():
    rows = [
        {
            "category": "concept",
            "required_tools_recalled": True,
            "has_extra_tool_call": False,
            "steps": 1,
            "source_hit": True,
            "source_top1": True,
        },
        {
            "category": "concept",
            "required_tools_recalled": False,
            "has_extra_tool_call": True,
            "steps": 2,
            "source_hit": False,
            "source_top1": False,
        },
        {
            "category": "signature",
            "required_tools_recalled": True,
            "has_extra_tool_call": False,
            "steps": 1,
            "source_hit": None,
            "source_top1": None,
        },
    ]

    summary = aggregate_metrics(rows)

    assert summary["overall"] == {
        "n": 3,
        "required_tool_recall": 2 / 3,
        "extra_call_rate": 1 / 3,
        "avg_steps": 4 / 3,
    }
    assert summary["by_category"]["concept"]["required_tool_recall"] == 0.5
    assert summary["concept_source"] == {
        "n": 2,
        "hit_rate": 0.5,
        "top1_rate": 0.5,
    }

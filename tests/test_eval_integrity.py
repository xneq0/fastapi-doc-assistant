"""评测集完整性测试（离线，只读文件）。

保证：schema 完整、id 唯一、可答题的 expected_source 存在且 evidence_quote
是该文档真实子串、不可答题 expected_source 为 null。防止评测集与文档脱节。
"""
import json
import re
from pathlib import Path

import pytest

DOCS = Path("docs")
EVAL = Path("eval")


def _load(name):
    return [json.loads(x) for x in (EVAL / name).read_text(encoding="utf-8").splitlines() if x.strip()]


DEV = _load("dev.jsonl")
TEST = _load("test.jsonl")
UNANS = _load("unanswerable.jsonl")
ANSWERABLE = DEV + TEST


def test_counts():
    assert len(ANSWERABLE) >= 40      # 目标 40+ 可答题
    assert len(UNANS) >= 10           # 目标 10+ 不可答题


def test_type_balance():
    from collections import Counter
    # 四类严格均衡：dev 每类 6、test 每类 4
    assert Counter(it["type"] for it in DEV) == {"semantic": 6, "task": 6, "keyword": 6, "error": 6}
    assert Counter(it["type"] for it in TEST) == {"semantic": 4, "task": 4, "keyword": 4, "error": 4}


def test_ids_unique():
    ids = [it["id"] for it in ANSWERABLE + UNANS]
    assert len(ids) == len(set(ids))


def test_dev_test_disjoint_ids():
    assert not ({it["id"] for it in DEV} & {it["id"] for it in TEST})


@pytest.mark.parametrize("item", ANSWERABLE, ids=[it["id"] for it in ANSWERABLE])
def test_answerable_evidence_is_substring(item):
    for key in ("question", "expected_source", "evidence_quote", "type"):
        assert item.get(key), f"{item['id']} 缺字段 {key}"
    doc = DOCS / item["expected_source"]
    assert doc.is_file(), f"{item['id']} 来源文件不存在: {item['expected_source']}"
    text = doc.read_text(encoding="utf-8")
    assert item["evidence_quote"] in text, f"{item['id']} 证据不是文档子串"


@pytest.mark.parametrize("item", UNANS, ids=[it["id"] for it in UNANS])
def test_unanswerable_has_no_source(item):
    assert item["expected_source"] is None
    assert item["type"].startswith("unanswerable")


@pytest.mark.parametrize("doc", sorted(DOCS.glob("tutorial/**/*.md")),
                         ids=lambda p: p.name)
def test_corpus_has_no_unprocessed_directives(doc):
    """生成语料不得残留未处理的 include / pymdownx 块标记。"""
    text = doc.read_text(encoding="utf-8")
    assert "{*" not in text and "*}" not in text, f"{doc} 残留 include 指令"
    assert "{!" not in text, f"{doc} 残留旧式 include"
    assert not re.search(r"(?m)^/{3,}", text), f"{doc} 残留 /// 或 //// 块标记"

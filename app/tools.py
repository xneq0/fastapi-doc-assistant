"""供后续 Agent 工具调用复用的普通 Python 函数。"""
import difflib
import inspect

import fastapi

from . import hybrid


def search_docs(query: str, k: int = 4) -> list[dict]:
    """用现有混合检索搜索文档片段，并返回稳定的工具字段。"""
    hits = hybrid.hybrid_search(query, k=k)
    return [
        {
            "source": hit["source"],
            "chunk_id": hit["id"],
            "text": hit["text"],
        }
        for hit in hits
    ]


def lookup_api(name: str) -> dict:
    """查询 FastAPI 的公开函数、类或方法签名。"""
    if not isinstance(name, str):
        return {"found": False, "message": "拒绝：名称必须是字符串。"}

    normalized = name.strip()
    parts = normalized.split(".")
    if (
        not normalized
        or len(parts) > 2
        or any(not part.isidentifier() or part.startswith("_") for part in parts)
    ):
        return {"found": False, "message": "拒绝：只允许一到两级公开属性名。"}

    public_top_level = [attr for attr in dir(fastapi) if not attr.startswith("_")]
    first = getattr(fastapi, parts[0], None)
    if first is None:
        if len(parts) == 2:
            return {
                "found": False,
                "message": "拒绝：只能从 fastapi 的公开对象开始查找。",
            }
        suggestions = difflib.get_close_matches(parts[0], public_top_level, n=3)
        result = {"found": False, "message": "fastapi 中没有这个公开对象。"}
        if suggestions:
            result["suggestions"] = suggestions
        return result

    parent = None
    obj = first
    if len(parts) == 2:
        parent = first
        public_children = [attr for attr in dir(parent) if not attr.startswith("_")]
        obj = getattr(parent, parts[1], None)
        if obj is None:
            suggestions = [
                f"{parts[0]}.{item}"
                for item in difflib.get_close_matches(parts[1], public_children, n=3)
            ]
            result = {"found": False, "message": "该公开对象没有这个属性。"}
            if suggestions:
                result["suggestions"] = suggestions
            return result

    if inspect.isclass(obj):
        kind = "class"
    elif parent is not None and inspect.isclass(parent) and callable(obj):
        kind = "method"
    elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
        kind = "function"
    else:
        return {"found": False, "message": "对象存在，但不是可查询的函数、类或方法。"}

    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return {"found": False, "message": "对象存在，但无法读取签名。"}

    doc = inspect.getdoc(obj) or ""
    first_paragraph = doc.split("\n\n", 1)[0].strip()
    return {
        "found": True,
        "name": normalized,
        "kind": kind,
        "signature": signature,
        "doc": first_paragraph,
    }

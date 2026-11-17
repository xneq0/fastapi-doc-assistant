"""最小工具调用循环；暂未接入 HTTP 接口。"""
import copy
import json

from . import config, llm, tools


SYSTEM = (
    "你是 FastAPI 文档助手。只依据工具返回的内容作答，不得补充未经工具验证的事实。"
    "引用 search_docs 的文档片段时，使用 [来源: source#chunk_id] 格式。"
    "如果工具结果没有提供足够依据，明确说明无法依据现有信息回答。"
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "用于回答 FastAPI 概念、用法和教程类问题；需要从项目文档中检索依据时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在 FastAPI 文档中检索的问题或关键词。",
                    },
                    "k": {
                        "type": "integer",
                        "description": "最多返回的文档片段数。",
                        "default": 4,
                        "minimum": 1,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_api",
            "description": (
                "用于查询某个具体 FastAPI 函数、类或方法的参数签名，例如 Query、Depends、"
                "HTTPException 或 FastAPI.get。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "FastAPI 公开函数、类或最多两级方法名。",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "search_docs": tools.search_docs,
    "lookup_api": tools.lookup_api,
}


def _tool_definitions(english_search_query: bool = True) -> list[dict]:
    """按实验开关返回工具说明，不修改默认全局定义。"""
    if not english_search_query:
        return TOOL_DEFINITIONS

    definitions = copy.deepcopy(TOOL_DEFINITIONS)
    search_function = definitions[0]["function"]
    search_function["description"] += (
        " 调用时必须用英文关键词填写 query，并尽量采用 FastAPI 官方文档中可能出现的英文术语；"
        "即使用户使用中文提问，也不要直接把中文问题作为 query。"
    )
    search_function["parameters"]["properties"]["query"]["description"] = (
        "英文检索关键词或短语，优先使用 FastAPI 文档中的英文术语。"
    )
    return definitions


def _summary(value, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _execute_tool(name: str, raw_arguments: str):
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return raw_arguments, {"error": "工具参数不是合法 JSON。"}

    if not isinstance(arguments, dict):
        return arguments, {"error": "工具参数必须是 JSON 对象。"}

    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        return arguments, {"error": f"未知工具：{name}"}

    try:
        return arguments, function(**arguments)
    except Exception as exc:  # noqa: BLE001 工具错误须反馈给模型而非中断循环
        return arguments, {"error": f"工具执行失败：{type(exc).__name__}"}


def run_agent(
    question: str,
    max_steps: int = 3,
    english_search_query: bool = True,
) -> dict:
    """让模型按需调用工具，直到生成最终回答或达到工具调用上限。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    trace = []
    steps = 0
    max_steps = max(0, max_steps)
    tool_definitions = _tool_definitions(english_search_query)

    while True:
        force_final = steps >= max_steps
        response = llm._client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=messages,
            tools=tool_definitions,
            tool_choice="none" if force_final else "auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return {
                "answer": message.content or "模型未返回可用的最终回答。",
                "trace": trace,
                "steps": steps,
            }

        if force_final:
            return {
                "answer": message.content or "已达到最大工具调用次数，无法基于现有信息完成回答。",
                "trace": trace,
                "steps": steps,
            }

        messages.append(message)
        for tool_call in tool_calls:
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            if steps >= max_steps:
                arguments = raw_arguments
                result = {"error": "已达到最大工具调用次数，工具未执行。"}
            else:
                arguments, result = _execute_tool(name, raw_arguments)
                steps += 1

            trace.append({
                "tool": name,
                "arguments": arguments,
                "result": _summary(result),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

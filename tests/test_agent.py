"""最小 Agent 工具循环的离线测试。"""
import os
import tempfile
from types import SimpleNamespace

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["DEEPSEEK_API_KEY"] = "test-key"

from app import agent  # noqa: E402


def _message(content=None, tool_calls=None):
    return SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or [])


def _tool_call(name, arguments, call_id="call-1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=function)


class _ScriptedCompletions:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(*messages):
    completions = _ScriptedCompletions(messages)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def test_run_agent_answers_without_tool(monkeypatch):
    client, completions = _fake_client(_message("直接回答"))
    monkeypatch.setattr(agent.llm, "_client", client)

    result = agent.run_agent("问题")

    assert result == {"answer": "直接回答", "trace": [], "steps": 0}
    assert "英文关键词" in completions.requests[0]["tools"][0]["function"]["description"]
    assert completions.requests[0]["tool_choice"] == "auto"


def test_english_search_query_is_default_and_can_be_disabled(monkeypatch):
    default_client, default = _fake_client(_message("default"))
    monkeypatch.setattr(agent.llm, "_client", default_client)
    agent.run_agent("问题")

    baseline_client, baseline = _fake_client(_message("baseline"))
    monkeypatch.setattr(agent.llm, "_client", baseline_client)
    agent.run_agent("问题", english_search_query=False)

    default_tools = default.requests[0]["tools"]
    baseline_tools = baseline.requests[0]["tools"]
    assert baseline_tools is agent.TOOL_DEFINITIONS
    assert "英文关键词" not in baseline_tools[0]["function"]["description"]
    assert "英文关键词" in default_tools[0]["function"]["description"]
    assert "英文检索关键词" in (
        default_tools[0]["function"]["parameters"]["properties"]["query"]["description"]
    )
    assert default_tools[1] == baseline_tools[1]
    assert agent.TOOL_DEFINITIONS == baseline_tools


def test_run_agent_calls_tool_then_answers(monkeypatch):
    call = _tool_call("lookup_api", '{"name": "Query"}')
    client, completions = _fake_client(_message(tool_calls=[call]), _message("最终回答"))
    monkeypatch.setattr(agent.llm, "_client", client)
    monkeypatch.setitem(agent._TOOL_FUNCTIONS, "lookup_api",
                        lambda name: {"found": True, "name": name})

    result = agent.run_agent("Query 的签名是什么？")

    assert result["answer"] == "最终回答"
    assert result["steps"] == 1
    assert result["trace"][0]["tool"] == "lookup_api"
    assert result["trace"][0]["arguments"] == {"name": "Query"}
    tool_message = completions.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"


def test_run_agent_unknown_tool_returns_error_to_model(monkeypatch):
    call = _tool_call("missing_tool", "{}")
    client, completions = _fake_client(_message(tool_calls=[call]), _message("无法调用该工具"))
    monkeypatch.setattr(agent.llm, "_client", client)

    result = agent.run_agent("问题")

    assert result["answer"] == "无法调用该工具"
    assert result["steps"] == 1
    assert "未知工具" in result["trace"][0]["result"]
    assert "未知工具" in completions.requests[1]["messages"][-1]["content"]


def test_run_agent_forces_final_answer_at_max_steps(monkeypatch):
    call = _tool_call("lookup_api", '{"name": "Query"}')
    client, completions = _fake_client(
        _message(tool_calls=[call]),
        _message("基于已有工具结果回答"),
    )
    monkeypatch.setattr(agent.llm, "_client", client)
    monkeypatch.setitem(agent._TOOL_FUNCTIONS, "lookup_api",
                        lambda name: {"found": True, "name": name})

    result = agent.run_agent("问题", max_steps=1)

    assert result["answer"] == "基于已有工具结果回答"
    assert result["steps"] == 1
    assert len(result["trace"]) == 1
    assert completions.requests[1]["tool_choice"] == "none"

"""Agent HTTP 接口的离线测试。"""
import os
import tempfile

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["DEEPSEEK_API_KEY"] = "test-key"

from fastapi.testclient import TestClient  # noqa: E402
from app import main  # noqa: E402


client = TestClient(main.app)


def test_agent_ask_returns_agent_result(monkeypatch):
    monkeypatch.setattr(main.config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        main.agent,
        "run_agent",
        lambda question: {
            "answer": f"回答：{question}",
            "steps": 1,
            "trace": [{"tool": "lookup_api", "arguments": {}, "result": "ok"}],
        },
    )

    response = client.post("/agent/ask", json={"question": "Query 的签名是什么？"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "回答：Query 的签名是什么？",
        "steps": 1,
        "trace": [{"tool": "lookup_api", "arguments": {}, "result": "ok"}],
    }


def test_agent_ask_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(
        main.agent,
        "run_agent",
        lambda question: (_ for _ in ()).throw(AssertionError("不应调用 Agent")),
    )

    response = client.post("/agent/ask", json={"question": ""})

    assert response.status_code == 422


def test_agent_ask_returns_safe_upstream_error(monkeypatch):
    monkeypatch.setattr(main.config, "DEEPSEEK_API_KEY", "test-key")

    def fail(_question):
        raise RuntimeError("包含不应暴露的信息")

    monkeypatch.setattr(main.agent, "run_agent", fail)
    response = client.post("/agent/ask", json={"question": "问题"})

    assert response.status_code == 502
    assert response.json() == {"detail": "DeepSeek 上游调用失败"}
    assert "不应暴露的信息" not in response.text


def test_agent_ask_reports_missing_key(monkeypatch):
    monkeypatch.setattr(main.config, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(
        main.agent,
        "run_agent",
        lambda question: (_ for _ in ()).throw(AssertionError("不应调用 Agent")),
    )

    response = client.post("/agent/ask", json={"question": "问题"})

    assert response.status_code == 503
    assert response.json() == {"detail": "DeepSeek API key 未配置"}

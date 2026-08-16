"""对话生成：DeepSeek（OpenAI 兼容接口）。"""
from openai import OpenAI
from . import config

_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def chat(system: str, user: str) -> str:
    resp = _client.chat.completions.create(
        model=config.DEEPSEEK_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

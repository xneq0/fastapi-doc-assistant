"""集中管理配置，从 .env 读取。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（app/ 的上一级），用于限制 /ingest 只能读项目内目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = "docs"

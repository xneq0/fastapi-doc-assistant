"""FastAPI 入口。启动: uvicorn app.main:app --reload"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import ingest, rag, vectorstore
from .config import PROJECT_ROOT

app = FastAPI(title="开发者文档问答助手（RAG 问答 + 混合检索）", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


class IngestRequest(BaseModel):
    dir_path: str = "docs"


def _safe_dir(dir_path: str) -> Path:
    """把 dir_path 限制在项目目录内，防止路径穿越/读取任意系统目录。"""
    p = Path(dir_path)
    target = (p if p.is_absolute() else PROJECT_ROOT / p).resolve()
    if PROJECT_ROOT not in target.parents and target != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="dir_path 必须位于项目目录内")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="目录不存在")
    return target


@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_db": vectorstore.count()}


@app.post("/ingest")
def do_ingest(req: IngestRequest):
    target = _safe_dir(req.dir_path)
    result = ingest.ingest_dir(str(target))
    return {"ingested": result, "total_chunks": vectorstore.count()}


@app.post("/ask")
def ask(req: AskRequest):
    return rag.answer(req.question, k=req.k)

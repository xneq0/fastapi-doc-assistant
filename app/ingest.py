"""文档切分（chunking）+ 入库。

MVP 用「按字符 + 重叠」的朴素切分。后续做深度点时，
这里是你换『按标题/父子分块』并对比效果的地方。
"""
import hashlib
from pathlib import Path
from . import vectorstore


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _source_name(p: Path, root: Path | None) -> str:
    """source 用『相对根目录的路径』而非纯文件名，
    避免不同子目录下的同名文件 ID/来源冲突。"""
    if root is not None:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            pass
    return p.name


def ingest_file(path: str, root: str | None = None) -> int:
    p = Path(path)
    if p.suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")

    source = _source_name(p, Path(root) if root else None)
    pieces = chunk_text(text)
    chunks = []
    for i, piece in enumerate(pieces):
        cid = hashlib.md5(f"{source}-{i}".encode()).hexdigest()
        chunks.append({"id": cid, "text": piece, "source": source, "chunk_index": i})
    if chunks:
        vectorstore.add_chunks(chunks)
    return len(chunks)


def ingest_dir(dir_path: str) -> dict:
    root = Path(dir_path)
    result = {}
    for f in sorted(root.glob("**/*")):
        if f.suffix in (".md", ".txt", ".pdf"):
            source = _source_name(f, root)
            result[source] = ingest_file(str(f), root=str(root))
    return result

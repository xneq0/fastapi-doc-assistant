"""可复现的官方文档同步脚本。

做法：把 FastAPI 官方仓库在钉定 commit 上做一次浅检出到本地缓存目录
（.fastapi_src/，已 gitignore），再从磁盘读取 tutorial markdown，解析其中
`{* ../../docs_src/xxx.py [ln[a:b]] [hl[...]] *}` 代码引入指令，把示例 Python
代码内联进 markdown，保存到 docs/ 下（保留相对路径）。

用本地检出而非逐个 raw 请求：避免 GitHub 限流、且一次抓取即钉死版本。
不手工复制任何代码；改文档集合只需改 DOC_PATHS 后重跑。
运行: ./.venv/bin/python -m scripts.sync_docs
"""
import re
import subprocess
import sys
from pathlib import Path

# ── 钉定来源（保证可复现）───────────────────────────────────────────
REPO_URL = "https://github.com/fastapi/fastapi.git"
REPO = "fastapi/fastapi"
PINNED_SHA = "a1fa70d4237d50aae6586a0d9b229df583463d21"  # master @ 抓取日
FETCH_DATE = "2026-08-17"
LICENSE = "MIT"
DOCS_BASE = "docs/en/docs"
DOC_URL_BASE = f"https://github.com/{REPO}/blob/{PINNED_SHA}/{DOCS_BASE}"

CACHE_DIR = Path(".fastapi_src")
OUT_DIR = Path("docs")

# 选定的 32 篇（相对 DOCS_BASE）。近义簇（params/body/dependencies）刻意保留。
DOC_PATHS = [
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/query-param-models.md",
    "tutorial/body.md",
    "tutorial/body-multiple-params.md",
    "tutorial/body-fields.md",
    "tutorial/body-nested-models.md",
    "tutorial/extra-data-types.md",
    "tutorial/cookie-params.md",
    "tutorial/header-params.md",
    "tutorial/response-model.md",
    "tutorial/extra-models.md",
    "tutorial/response-status-code.md",
    "tutorial/request-forms.md",
    "tutorial/request-files.md",
    "tutorial/request-forms-and-files.md",
    "tutorial/handling-errors.md",
    "tutorial/path-operation-configuration.md",
    "tutorial/encoder.md",
    "tutorial/body-updates.md",
    "tutorial/dependencies/index.md",
    "tutorial/dependencies/classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md",
    "tutorial/dependencies/dependencies-with-yield.md",
    "tutorial/bigger-applications.md",
    "tutorial/background-tasks.md",
    "tutorial/metadata.md",
    "tutorial/static-files.md",
    "tutorial/testing.md",
]

INCLUDE_RE = re.compile(r"\{\*\s*(?P<path>\S+)(?P<args>[^*]*)\*\}")
HEADING_ANCHOR_RE = re.compile(r"\s*\{\s*#[^}]+\}\s*$")
LN_RE = re.compile(r"ln\[(\d+):(\d+)\]")
LANG_BY_EXT = {".py": "python", ".sh": "console", ".txt": "text",
               ".json": "json", ".html": "html", ".yaml": "yaml", ".yml": "yaml"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(CACHE_DIR), *args],
                          capture_output=True, text=True)


def ensure_checkout() -> bool:
    """在 PINNED_SHA 上严格浅检出仓库到 CACHE_DIR（幂等）。

    强制版本锁定：只按 PINNED_SHA fetch/checkout，失败即报错退出；
    绝不回退到 master；最终 HEAD 必须严格等于 PINNED_SHA，否则失败。
    """
    head = _git("rev-parse", "HEAD").stdout.strip() if (CACHE_DIR / ".git").exists() else ""
    if head == PINNED_SHA:
        print(f"复用已检出的 {PINNED_SHA[:10]}")
        return True
    CACHE_DIR.mkdir(exist_ok=True)
    if not (CACHE_DIR / ".git").exists():
        _git("init", "-q")
        _git("remote", "add", "origin", REPO_URL)
    print(f"浅检出 {REPO}@{PINNED_SHA[:10]} ...")
    r = _git("fetch", "-q", "--depth", "1", "origin", PINNED_SHA)
    if r.returncode != 0:
        print(f"✗ 按钉定 SHA fetch 失败（不回退 master）: {r.stderr.strip()[:200]}")
        return False
    co = _git("checkout", "-q", "FETCH_HEAD")
    if co.returncode != 0:
        print(f"✗ checkout 失败: {co.stderr.strip()[:200]}")
        return False
    got = _git("rev-parse", "HEAD").stdout.strip()
    if got != PINNED_SHA:
        print(f"✗ HEAD={got} 与钉定 SHA={PINNED_SHA} 不一致，中止。")
        return False
    return True


def read(rel_from_root: str) -> str | None:
    p = CACHE_DIR / rel_from_root
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _extract_src_path(raw_path: str) -> str | None:
    """把 '../../docs_src/x/y.py' 归一到仓库根的 'docs_src/x/y.py'。"""
    m = re.search(r"docs_src/.+", raw_path)
    return m.group(0) if m else None


def inline_includes(md: str, stats: dict) -> str:
    def repl(match: re.Match) -> str:
        src_path = _extract_src_path(match.group("path"))
        args = match.group("args") or ""
        if not src_path:
            stats["skipped"] += 1
            return match.group(0)
        code = read(src_path)
        if code is None:
            stats["missing"] += 1
            return f"```\n# (示例代码缺失: {src_path})\n```"
        ln = LN_RE.search(args)
        if ln:  # ln[a:b] 只取指定行范围
            lines = code.splitlines()
            code = "\n".join(lines[int(ln.group(1)) - 1:int(ln.group(2))])
        lang = LANG_BY_EXT.get(Path(src_path).suffix, "")
        stats["inlined"] += 1
        return f"```{lang}\n{code.rstrip()}\n```"

    return INCLUDE_RE.sub(repl, md)


def clean_markdown(md: str) -> str:
    out = []
    for line in md.splitlines():
        if line.startswith("#"):
            line = HEADING_ANCHOR_RE.sub("", line)
        # pymdownx 块标记：`/// note`（提示）与 `//// tab | 标题`（标签页）等，3 个及以上斜杠
        m = re.match(r"^/{3,}\s*(\w+)?\s*(?:\|\s*(.*?))?\s*$", line)
        if m:
            kind, title = m.group(1), m.group(2)
            if title:
                out.append(f"**{title.strip()}:**")    # 标签页/details 的标题
            elif kind:
                out.append(f"**{kind.capitalize()}:**")  # note/tip/warning 等提示标签
            else:
                out.append("")                           # 光秃秃的收尾行 /// 或 ////
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"</?dfn[^>]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# 生成后若仍残留这些标记，说明有未处理的 include / 模板指令
RESIDUE_PATTERNS = {
    "include `{* *}`": re.compile(r"\{\*.*?\*\}"),
    "old-include `{! !}`": re.compile(r"\{!.*?!\}"),
    "pymdownx 块 `///`": re.compile(r"(?m)^/{3,}"),
}


def find_residue(text: str) -> list[str]:
    return [name for name, pat in RESIDUE_PATTERNS.items() if pat.search(text)]


def main() -> int:
    if not ensure_checkout():
        return 1

    # 第一遍：全内存处理并收集错误，任何问题都中止、不写盘（原子性）
    stats = {"inlined": 0, "missing": 0, "skipped": 0}
    processed_docs = []       # (rel, text)
    errors = []
    for rel in DOC_PATHS:
        md = read(f"{DOCS_BASE}/{rel}")
        if md is None:
            errors.append(f"缺文档: {rel}")
            continue
        before = (stats["missing"], stats["skipped"])
        text = clean_markdown(inline_includes(md, stats))
        if stats["missing"] > before[0]:
            errors.append(f"include 示例代码缺失: {rel}")
        if stats["skipped"] > before[1]:
            errors.append(f"include 指令无法解析: {rel}")
        residue = find_residue(text)
        if residue:
            errors.append(f"残留未处理标记 {residue}: {rel}")
        processed_docs.append((rel, text))

    if errors:
        print("✗ 同步失败，未写入任何文件：")
        for e in errors:
            print("   -", e)
        return 1

    # 第二遍：确认无误后再落盘
    OUT_DIR.mkdir(exist_ok=True)
    manifest = []
    for rel, text in processed_docs:
        out_path = OUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        manifest.append((rel, f"{DOC_URL_BASE}/{rel}"))
        print(f"  ✓ {rel}")

    lines = [
        "# 文档来源（SOURCES）", "",
        f"- 来源仓库: https://github.com/{REPO}",
        f"- 钉定 commit: `{PINNED_SHA}`",
        f"- 抓取日期: {FETCH_DATE}",
        f"- 许可证: {LICENSE}（见 LICENSE-FASTAPI.txt）",
        "- 生成方式: `./.venv/bin/python -m scripts.sync_docs`（含 docs_src 示例代码内联）",
        "", "| 本地路径 | 官方来源 |", "|----------|----------|",
    ]
    lines += [f"| docs/{rel} | [{rel}]({url}) |" for rel, url in manifest]
    # 溯源文件写到项目根目录，不放进 docs/（否则会被当成语料入库）
    Path("SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lic = read("LICENSE")
    if lic:
        Path("LICENSE-FASTAPI.txt").write_text(lic, encoding="utf-8")

    print(f"\n完成: {len(manifest)} 篇 | 内联代码 {stats['inlined']} 处 | "
          f"缺失 {stats['missing']} | 跳过 {stats['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

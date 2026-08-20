"""DeepSeek rerank：只对补充候选单元排序（不碰 base，不改 /ask）。

约束落点：
- 每个候选输入含 unit_id、anchor 文本、neighbor 文本、方向、anchor 原始 hybrid 名次；
- 只让模型判断"哪个补充单元最有助于回答问题"；
- 严格 JSON: {"ranked_unit_ids": [...]}，必须是全部候选 unit_id 的**完整排列**；
- 校验：类型均为 string / 无非法 / 无重复 / 无缺失 / 数量一致；任一不满足 → 回退确定性顺序；
- temperature=0；缓存键直接 hash 实际 system+user prompt（覆盖 model、prompt 版本、question、
  unit_id、anchor_rank、direction/delta、anchor/neighbor 文本）；
- 保存原始返回、解析结果、输入/输出 token、延迟、错误原因；复现以缓存为准。

延迟创建 client：模块导入、cache-only、有效缓存读取均**不需要 API key**；
只有真正发起调用（_call）时才实例化 OpenAI client。
"""
import json
import hashlib
import time
from pathlib import Path

from . import config

PROMPT_VERSION = "v1"
CACHE_DIR = Path("eval/rerank_cache")

SYSTEM = (
    "You are a retrieval reranker. You are given a QUESTION and a list of candidate "
    "SUPPLEMENT UNITS. Each unit is an anchor passage plus one adjacent neighbor passage "
    "that could be appended to an existing answer context. Rank ALL units from most to "
    "least helpful for answering the question. You must return STRICT JSON of the exact form "
    '{"ranked_unit_ids": ["<id>", ...]} containing EVERY candidate unit_id EXACTLY ONCE, '
    "no extra text."
)

_TEL_KEYS = ("in_tokens", "out_tokens", "latency", "api_calls")


def build_user_prompt(question: str, cands: list[dict]) -> str:
    lines = [f"QUESTION: {question}", "", "CANDIDATE SUPPLEMENT UNITS:"]
    for c in cands:
        direction = "next(idx+1)" if c["delta"] > 0 else "prev(idx-1)"
        lines += [
            f"- unit_id: {c['unit_id']}",
            f"  anchor_hybrid_rank: {c['anchor_rank']}  direction: {direction}",
            f"  anchor_text: {c['anchor_text'][:500]}",
            f"  neighbor_text: {c['neighbor_text'][:500]}",
        ]
    lines += ["", 'Return STRICT JSON: {"ranked_unit_ids": [all unit_id values, each once]}']
    return "\n".join(lines)


def cache_key(question: str, cands: list[dict]) -> str:
    """直接 hash 实际发送的 system+user prompt，加上 model 与 prompt 版本。

    user prompt 已包含 question、每个候选的 unit_id、anchor_rank、direction、
    anchor/neighbor 文本，因此任何影响提示词的字段变化都会改变缓存键。
    """
    user = build_user_prompt(question, cands)
    payload = "\n\x1e\n".join([config.DEEPSEEK_CHAT_MODEL, PROMPT_VERSION, SYSTEM, user])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_permutation(order, valid_ids):
    """校验 order 是否为 valid_ids 的完整排列。类型安全：先确保元素均为 str，
    再做集合运算，避免 dict/list 等不可 hash 值触发 TypeError。"""
    valid = list(valid_ids)
    if not isinstance(order, list):
        return False, "not_list"
    if any(not isinstance(x, str) for x in order):
        return False, "non_string_id"
    if len(order) != len(valid):
        return False, "count_mismatch"
    if len(set(order)) != len(order):
        return False, "duplicate_ids"
    if set(order) != set(valid):
        return False, "id_mismatch"          # 覆盖非法(多余)与缺失
    return True, None


def validate(raw: str, cands: list[dict]):
    """返回 (order|None, error|None)。任何不满足完整排列的情况都判失败。"""
    valid = [c["unit_id"] for c in cands]
    try:
        obj = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return None, f"json_error:{type(e).__name__}"
    if not isinstance(obj, dict) or "ranked_unit_ids" not in obj:
        return None, "missing_key"
    ok, err = _check_permutation(obj["ranked_unit_ids"], valid)
    return (obj["ranked_unit_ids"] if ok else None), (None if ok else err)


_client = None


def _get_client():
    """延迟创建 OpenAI client：只有真正发起调用时才实例化（需要 key）。"""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


def _call(system: str, user: str):
    t0 = time.time()
    resp = _get_client().chat.completions.create(
        model=config.DEEPSEEK_CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    dt = time.time() - t0
    u = resp.usage
    return resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens, dt


def _read_cache(cfile: Path, cands: list[dict]):
    """读缓存并做**完整语义校验**（针对当前候选 unit_id）。返回 (rec|None, cache_error|None)。

    任何 IO/JSON/字段缺失/order 非合法排列都视为缓存损坏（rec=None），
    由调用方按缓存缺失处理（allow_api=False 时确定性回退，绝不调用 API）。
    """
    try:
        rec = json.loads(cfile.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"cache_error:{type(e).__name__}"
    for f in ("order", "parse_ok", "error", "telemetry"):
        if f not in rec:
            return None, "cache_error:missing_field"
    tel = rec["telemetry"]
    if not isinstance(tel, dict) or any(k not in tel for k in _TEL_KEYS):
        return None, "cache_error:bad_telemetry"
    # 缓存命中后按当前候选重新做完整排列校验（类型/重复/缺失/多余）
    ok, perr = _check_permutation(rec["order"], [c["unit_id"] for c in cands])
    if not ok:
        return None, f"cache_error:{perr}"
    return rec, None


def rerank_units(question: str, cands: list[dict], use_cache: bool = True,
                 chat_fn=None, allow_api: bool = False) -> dict:
    """对候选单元 rerank。返回 {order|None, parse_ok, error, telemetry, cached, cache_error}。

    order=None 表示解析失败/API 异常/API 被禁用/无候选，调用方应回退确定性顺序。

    硬开关 allow_api（默认 False）：真实 DeepSeek 调用只在 allow_api=True 时才允许。
    - 有效缓存：始终可读，不受 allow_api 影响（且会按当前候选重新语义校验）。
    - 缓存缺失/损坏/语义无效 且 allow_api=False：返回 error="api_disabled"、api_calls=0，
      **绝不自动调用** DeepSeek；cache_error 记录原因。
    - 传入 chat_fn 测试桩时视为"非真实 API"，不受 allow_api 闸门限制（仅供离线测试）。

    其它：API 异常 → error="api_error:<类型>"（不含 key），api_calls=1，不写缓存。
    """
    empty_tel = {"in_tokens": 0, "out_tokens": 0, "latency": 0.0, "api_calls": 0}
    if not cands:
        return {"order": [], "parse_ok": True, "error": None, "telemetry": empty_tel,
                "cached": False, "cache_error": None}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = CACHE_DIR / f"{cache_key(question, cands)}.json"
    cache_error = None
    if use_cache and cfile.exists():
        rec, cache_error = _read_cache(cfile, cands)
        if rec is not None:
            return {"order": rec["order"], "parse_ok": rec["parse_ok"], "error": rec["error"],
                    "telemetry": rec["telemetry"], "cached": True, "cache_error": None}
        # 缓存损坏/语义无效 → 视为 miss，落到下面的闸门判定

    # 硬闸门：真实 API（chat_fn is None）在 allow_api=False 时一律禁止，绝不自动调用。
    # 测试桩（chat_fn 不为空）不算真实 API，可放行。
    if chat_fn is None and not allow_api:
        return {"order": None, "parse_ok": False, "error": "api_disabled",
                "telemetry": dict(empty_tel), "cached": False,
                "cache_error": cache_error or "cache_miss"}

    system, user = SYSTEM, build_user_prompt(question, cands)
    t0 = time.time()
    try:
        raw, in_tok, out_tok, dt = (chat_fn or _call)(system, user)
    except Exception as e:  # noqa: BLE001  网络/超时/401/429/5xx 等，只记类名，绝不含 key
        tel = {"in_tokens": 0, "out_tokens": 0, "latency": round(time.time() - t0, 3), "api_calls": 1}
        return {"order": None, "parse_ok": False, "error": f"api_error:{type(e).__name__}",
                "telemetry": tel, "cached": False, "cache_error": cache_error}

    order, error = validate(raw, cands)
    tel = {"in_tokens": in_tok, "out_tokens": out_tok, "latency": round(dt, 3), "api_calls": 1}
    rec = {"question": question, "unit_ids": [c["unit_id"] for c in cands],
           "raw": raw, "order": order, "parse_ok": order is not None,
           "error": error, "telemetry": tel}
    cfile.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"order": order, "parse_ok": order is not None, "error": error,
            "telemetry": tel, "cached": False, "cache_error": cache_error}

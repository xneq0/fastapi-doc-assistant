"""方案 A 流水线纯函数单测（离线：不触网、不读真实库）。"""
import json
import os
import tempfile

import pytest

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()  # 隔离，避免导入时连真实库


@pytest.fixture(autouse=True)
def _isolate_rerank_cache(tmp_path, monkeypatch):
    """兜底：所有测试的 rerank 缓存目录都指向临时目录，
    默认 pytest 绝不创建/覆盖/删除真实 eval/rerank_cache。个别测试可再自行覆盖。"""
    from app import rerank as _rr
    monkeypatch.setattr(_rr, "CACHE_DIR", tmp_path / "_rerank_cache")

from app.retrieval_pipeline import (  # noqa: E402
    expand_adjacency, order_a0, order_ab, order_aa, cap,
    make_windows, score_windows, select_windows, dedup_order,
)


def _anchor(source, idx, score):
    return {"id": f"{source}#{idx}", "source": source, "chunk_index": idx, "score": score}


def _lookup(source, n):
    return {(source, i): {"id": f"{source}#{i}", "source": source,
                          "chunk_index": i, "text": f"{source} chunk {i}"} for i in range(n)}


def test_expand_brings_in_adjacent_evidence_chunk():
    # dev20 型：锚点命中 #1，证据在未被检索的 #2 → 扩展后 #2 必须进池
    lookup = _lookup("p", 5)
    pool = expand_adjacency([_anchor("p", 1, 1.0)], lookup, w=1)
    ids = {c["id"] for c in pool}
    assert "p#2" in ids and "p#0" in ids and "p#1" in ids
    # 截断后仍在最终上下文里
    final = cap(order_a0(pool), per_source=3, k_final=3)
    assert any(c["source"] == "p" and c["chunk_index"] == 2 for c in final)


def test_expand_clamps_and_no_cross_source():
    lookup = _lookup("p", 3)          # 只有 #0..#2
    pool = expand_adjacency([_anchor("p", 0, 1.0)], lookup, w=2)
    idxs = sorted(c["chunk_index"] for c in pool)
    assert idxs == [0, 1, 2]          # 不会造出 #-2/#-1/#3/#4
    assert all(c["source"] == "p" for c in pool)


def test_expand_dedup_and_max_inheritance():
    lookup = _lookup("p", 4)
    # 两个锚点 #1(0.9) 与 #3(1.0) 都会覆盖 #2；#2 取较大的继承分
    pool = expand_adjacency([_anchor("p", 1, 0.9), _anchor("p", 3, 1.0)], lookup, w=1)
    c2 = next(c for c in pool if c["chunk_index"] == 2)
    assert len({c["id"] for c in pool}) == len(pool)         # 无重复 id
    assert abs(c2["inherited"] - 1.0 * 0.9) < 1e-9           # max(0.9*0.9, 1.0*0.9)=0.9


def test_order_a0_desc_by_inherited():
    pool = [
        {"id": "a", "source": "s", "chunk_index": 0, "inherited": 0.2},
        {"id": "b", "source": "s", "chunk_index": 1, "inherited": 0.9},
    ]
    assert [c["id"] for c in order_a0(pool)] == ["b", "a"]


def test_order_ab_lets_high_cosine_climb():
    # 继承分 a>b>c（A0 下 c 垫底）；但只有 c 与 query 同向 → RRF 融合后 c 升到第一
    pool = [
        {"id": "a", "source": "s", "chunk_index": 0, "inherited": 0.9},
        {"id": "b", "source": "s", "chunk_index": 1, "inherited": 0.5},
        {"id": "c", "source": "s", "chunk_index": 2, "inherited": 0.1},
    ]
    qv = [1.0, 0.0]
    emb = {"a": [0.0, 1.0], "b": [0.0, 1.0], "c": [1.0, 0.0]}  # 仅 c 与 query 同向
    assert order_a0(pool)[-1]["id"] == "c"            # A0 下 c 垫底
    res = [x["id"] for x in order_ab(pool, qv, emb)]  # A-b 下 cosine 把 c 从垫底抬过 b
    assert res == ["a", "c", "b"]
    assert res.index("c") < res.index("b")


def test_order_aa_single_point():
    pool = [
        {"id": "a", "source": "s", "chunk_index": 0, "inherited": 1.0},
        {"id": "b", "source": "s", "chunk_index": 1, "inherited": 0.0},
    ]
    qv = [1.0, 0.0]
    emb = {"a": [0.0, 1.0], "b": [1.0, 0.0]}
    # α=0.5：a 继承满分、b cosine 满分 → 打平，按确定性 tie-break (source,idx) → a 在前
    assert [c["id"] for c in order_aa(pool, qv, emb, alpha=0.5)] == ["a", "b"]


def test_cap_enforces_per_source_and_global():
    ordered = [
        {"id": "a0", "source": "a", "chunk_index": 0},
        {"id": "a1", "source": "a", "chunk_index": 1},
        {"id": "a2", "source": "a", "chunk_index": 2},
        {"id": "b0", "source": "b", "chunk_index": 0},
    ]
    out = cap(ordered, per_source=2, k_final=10)
    assert [c["id"] for c in out] == ["a0", "a1", "b0"]      # 每源最多 2
    assert len(cap(ordered, per_source=10, k_final=2)) == 2  # 全局上限 2


# ── 窗口级排序（下一轮实验）单测 ────────────────────────────────────
def _anc(source, idx, score, rank):
    return {"id": f"{source}#{idx}", "source": source, "chunk_index": idx, "score": score}


def test_make_windows_clamps_to_existing():
    lookup = _lookup("p", 3)                       # #0..#2
    wins = make_windows([_anc("p", 0, 1.0, 0)], lookup, w=1)
    assert len(wins) == 1
    assert [c["chunk_index"] for c in wins[0]["chunks"]] == [0, 1]   # #-1 不存在被 clamp


def test_score_windows_ws_max_uses_best_member():
    # dev20 型：证据在 p#2，但窗内最高分成员是锚点 p#1；整窗仍被托起并带上 p#2
    lookup = _lookup("p", 3)                        # p 的窗 {0,1,2}
    lookup.update(_lookup("q", 3))                  # q 的窗 {0,1,2}
    anchors = [_anc("p", 1, 0.1, 0), _anc("q", 1, 0.9, 1)]  # p 锚点分低、q 锚点分高
    wins = make_windows(anchors, lookup, w=1)
    qv = [1.0, 0.0]
    emb = {f"p#{i}": [0.0, 1.0] for i in range(3)}
    emb["p#1"] = [1.0, 0.0]                          # 仅 p#1（锚点）与 query 同向
    emb.update({f"q#{i}": [0.2, 0.9] for i in range(3)})
    ordered = score_windows(wins, qv, emb, "WS-max")
    assert ordered[0]["anchor_source"] == "p"       # p 窗因 p#1 高分排第一
    final = dedup_order(select_windows(ordered, k_final=6, s_per_source=3))
    assert any(c["id"] == "p#2" for c in final)      # 证据邻居 p#2 被整窗带入


def test_select_windows_atomic_skip_over_budget():
    # 预算不足以整窗放入 → 跳过整窗，继续放得下的后续窗；不部分保留
    big = {"anchor_source": "a", "anchor_idx": 1, "anchor_rank": 0,
           "chunks": [{"id": "a1", "source": "a", "chunk_index": 1},
                      {"id": "a2", "source": "a", "chunk_index": 2},
                      {"id": "a3", "source": "a", "chunk_index": 3}],
           "chunk_ids": ["a1", "a2", "a3"]}
    small = {"anchor_source": "b", "anchor_idx": 0, "anchor_rank": 1,
             "chunks": [{"id": "b0", "source": "b", "chunk_index": 0}], "chunk_ids": ["b0"]}
    sel = select_windows([big, small], k_final=2, s_per_source=3)
    assert [w["anchor_source"] for w in sel] == ["b"]   # big(3) 被跳过，small(1) 入选
    assert len(dedup_order(sel)) == 2 - 1               # 最终 1 个 chunk，未超 K_final


def test_select_windows_per_source_cap():
    ws = [{"anchor_source": "a", "anchor_idx": i, "anchor_rank": i,
           "chunks": [{"id": f"a{i}", "source": "a", "chunk_index": i}],
           "chunk_ids": [f"a{i}"]} for i in range(5)]
    sel = select_windows(ws, k_final=10, s_per_source=2)
    assert len(sel) == 2                                # 同源最多 2 个窗


def test_dedup_order_dedup_overlapping_windows():
    w1 = {"anchor_source": "p", "anchor_idx": 1, "anchor_rank": 0,
          "chunks": [{"id": "p0", "source": "p", "chunk_index": 0},
                     {"id": "p1", "source": "p", "chunk_index": 1},
                     {"id": "p2", "source": "p", "chunk_index": 2}], "chunk_ids": ["p0", "p1", "p2"]}
    w2 = {"anchor_source": "p", "anchor_idx": 2, "anchor_rank": 1,
          "chunks": [{"id": "p1", "source": "p", "chunk_index": 1},
                     {"id": "p2", "source": "p", "chunk_index": 2},
                     {"id": "p3", "source": "p", "chunk_index": 3}], "chunk_ids": ["p1", "p2", "p3"]}
    final = dedup_order([w1, w2])
    assert [c["id"] for c in final] == ["p0", "p1", "p2", "p3"]   # 重叠去重，无重复


# ── 基线保留式补充检索（第一阶段，无 rerank）单测 ──────────────────
from app.retrieval_pipeline import supplement_base  # noqa: E402


def _ch(source, idx, text="x" * 40):
    return {"id": f"{source}#{idx}", "source": source, "chunk_index": idx, "text": text}


def _lk(source, n, text="x" * 40):
    return {(source, i): _ch(source, i, text) for i in range(n)}


def test_supplement_base_is_frozen_prefix():
    # base 顺序/内容不变，补充只追加在后
    base = [_ch("a", 0), _ch("b", 0)]
    anchors = [{"id": "a#0", "source": "a", "chunk_index": 0}]
    lookup = _lk("a", 3)
    lookup[("b", 0)] = _ch("b", 0)
    r = supplement_base(base, anchors, lookup, "N-next", budget_factor=0.5)
    assert r["final"][:2] == base                       # 前缀完全冻结
    assert all(c["id"] not in {"a#0", "b#0"} for c in r["supplements"])  # 不重复 base


def test_supplement_anchor_in_base_adds_only_neighbor():
    base = [_ch("a", 1)]                                 # anchor a#1 已在 base
    anchors = [{"id": "a#1", "source": "a", "chunk_index": 1}]
    lookup = _lk("a", 4)
    r = supplement_base(base, anchors, lookup, "N-next", budget_factor=10)
    ids = [c["id"] for c in r["supplements"]]
    assert ids == ["a#2"]                                # 只补邻居，不重复 anchor


def test_supplement_anchor_not_in_base_is_atomic_pair():
    base = [_ch("z", 0)]                                 # base 与 anchor 无关
    anchors = [{"id": "a#1", "source": "a", "chunk_index": 1}]
    lookup = _lk("a", 4)
    lookup[("z", 0)] = _ch("z", 0)
    r = supplement_base(base, anchors, lookup, "N-next", budget_factor=10)
    ids = [c["id"] for c in r["supplements"]]
    assert ids == ["a#1", "a#2"]                         # anchor+neighbor 原子加入


def test_supplement_budget_skips_oversized_unit_but_continues():
    # 预算只够 1 块；第一个单元(anchor不在base=2块)放不下→跳过；后一个(1块)可加入
    base = [_ch("z", 0, "x" * 40)]                       # base ≈ 10 tok, budget=0.5×=5
    anchors = [
        {"id": "a#1", "source": "a", "chunk_index": 1},  # 不在base → 2块单元, 太大
        {"id": "z#0", "source": "z", "chunk_index": 0},  # 在base → 只补 z#1, 1块
    ]
    lookup = {("a", 1): _ch("a", 1), ("a", 2): _ch("a", 2),
              ("z", 0): _ch("z", 0), ("z", 1): _ch("z", 1, "x" * 12)}  # z#1 ≈ 3 tok
    r = supplement_base(base, anchors, lookup, "N-next", budget_factor=0.5)
    ids = [c["id"] for c in r["supplements"]]
    assert "a#1" not in ids and "a#2" not in ids          # 大单元被跳过
    assert ids == ["z#1"]                                  # 小单元仍被加入


def test_supplement_deterministic_order():
    # 两个锚点，确定性序：先按 anchor_rank
    base = [_ch("z", 0)]
    anchors = [
        {"id": "b#1", "source": "b", "chunk_index": 1},   # rank0
        {"id": "a#1", "source": "a", "chunk_index": 1},   # rank1
    ]
    lookup = {("z", 0): _ch("z", 0)}
    lookup.update(_lk("a", 3)); lookup.update(_lk("b", 3))
    r = supplement_base(base, anchors, lookup, "N-next", budget_factor=100)
    ids = [c["id"] for c in r["supplements"]]
    assert ids == ["b#1", "b#2", "a#1", "a#2"]            # rank0 的 b 先于 rank1 的 a


def test_supplement_nwin_two_neighbors():
    base = [_ch("z", 0)]
    anchors = [{"id": "a#2", "source": "a", "chunk_index": 2}]
    lookup = {("z", 0): _ch("z", 0)}; lookup.update(_lk("a", 5))
    r = supplement_base(base, anchors, lookup, "N-win", budget_factor=100)
    ids = sorted(c["id"] for c in r["supplements"])
    assert ids == ["a#1", "a#2", "a#3"]                  # idx-1, idx(anchor), idx+1


# ── build_candidates 复合 unit_id ───────────────────────────────────
from app.retrieval_pipeline import build_candidates  # noqa: E402


def test_build_candidates_composite_unit_id():
    base = [_ch("z", 0)]
    lookup = {("z", 0): _ch("z", 0)}
    lookup.update(_lk("a", 5))
    # 两个不同 anchor 指向同一 neighbor a#2 → 不同 unit_id
    anchors = [{"id": "a#1", "source": "a", "chunk_index": 1},
               {"id": "a#3", "source": "a", "chunk_index": 3}]
    cands = build_candidates(base, anchors, lookup, "N-win")
    uids = [c["unit_id"] for c in cands]
    assert "a#1->a#2" in uids and "a#3->a#2" in uids       # 同 neighbor 不同复合 id
    assert len(uids) == len(set(uids))                     # 复合 id 唯一


def test_supplement_rerank_order_pulls_evidence_into_budget():
    # dev20 型：证据单元的 anchor 排名靠后，纯默认序会被预算挤出；
    # rerank 把证据 unit_id 排到最前 → 进入 0.5×base 预算
    base = [_ch("z", 0, "x" * 200)]                         # base≈50tok, budget≈25
    lookup = {("z", 0): _ch("z", 0, "x" * 200)}
    lookup.update(_lk("hot", 4, "x" * 40))                 # 热门源，anchor rank 靠前
    lookup.update(_lk("p", 4, "x" * 40))                   # 证据源，anchor rank 靠后
    anchors = [{"id": "hot#1", "source": "hot", "chunk_index": 1},   # rank0
               {"id": "p#1", "source": "p", "chunk_index": 1}]       # rank1（证据 p#2 在此邻）
    # 默认序：hot 的邻居先占预算
    d = supplement_base(base, anchors, lookup, "N-win", budget_factor=0.5)
    # rerank：把证据单元 p#1->p#2 放最前
    ev_uid = "p#1->p#2"
    order = [ev_uid] + [c["unit_id"] for c in d["candidates"] if c["unit_id"] != ev_uid]
    r = supplement_base(base, anchors, lookup, "N-win", budget_factor=0.5, order_unit_ids=order)
    assert any(c["id"] == "p#2" for c in r["supplements"])  # 证据邻居被 rerank 提入预算


# ── rerank JSON 校验（离线，stub chat_fn）─────────────────────────────
from app import rerank  # noqa: E402


def _cands3():
    return [{"unit_id": "a->b", "anchor_text": "t1", "neighbor_text": "n1", "delta": 1, "anchor_rank": 0},
            {"unit_id": "c->d", "anchor_text": "t2", "neighbor_text": "n2", "delta": -1, "anchor_rank": 1},
            {"unit_id": "e->f", "anchor_text": "t3", "neighbor_text": "n3", "delta": 1, "anchor_rank": 2}]


def test_validate_accepts_full_permutation():
    order, err = rerank.validate('{"ranked_unit_ids": ["e->f","a->b","c->d"]}', _cands3())
    assert err is None and order == ["e->f", "a->b", "c->d"]


def test_validate_rejects_duplicate_missing_illegal_count():
    c = _cands3()
    assert rerank.validate('{"ranked_unit_ids": ["a->b","a->b","c->d"]}', c)[0] is None  # dup
    assert rerank.validate('{"ranked_unit_ids": ["a->b","c->d"]}', c)[0] is None          # missing/count
    assert rerank.validate('{"ranked_unit_ids": ["a->b","c->d","X->Y"]}', c)[0] is None    # illegal
    assert rerank.validate('not json', c)[0] is None                                       # bad json
    assert rerank.validate('{"foo": 1}', c)[0] is None                                     # missing key


def test_rerank_units_fallback_on_bad_json(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)     # 隔离缓存，勿写真实 eval/rerank_cache
    c = _cands3()
    stub = lambda s, u: ("garbage", 10, 5, 0.01)            # noqa: E731
    r = rerank.rerank_units("q", c, use_cache=False, chat_fn=stub)
    assert r["order"] is None and r["parse_ok"] is False and r["error"] is not None
    assert r["telemetry"]["api_calls"] == 1


def test_rerank_units_ok_with_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)     # 隔离缓存
    c = _cands3()
    stub = lambda s, u: ('{"ranked_unit_ids": ["c->d","e->f","a->b"]}', 20, 8, 0.02)  # noqa: E731
    r = rerank.rerank_units("q", c, use_cache=False, chat_fn=stub)
    assert r["parse_ok"] and r["order"] == ["c->d", "e->f", "a->b"]


# ── rerank 容错：API 异常 + 坏缓存（离线）────────────────────────────
import httpx as _httpx  # noqa: E402


def test_rerank_api_exception_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)

    def boom(system, user):
        raise _httpx.ConnectError("network down")   # 模拟网络异常
    r = rerank.rerank_units("q", _cands3(), use_cache=True, chat_fn=boom)
    assert r["order"] is None and r["parse_ok"] is False
    assert r["error"] == "api_error:ConnectError"
    assert r["telemetry"]["api_calls"] == 1 and r["telemetry"]["latency"] >= 0
    # API 异常不写缓存（瞬时错误，可重试）
    assert not any(tmp_path.iterdir())


def test_rerank_api_error_message_excludes_key(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    secret = "sk-REALKEY123"

    def boom(system, user):
        raise RuntimeError(f"401 unauthorized for key {secret}")
    r = rerank.rerank_units("q", _cands3(), use_cache=False, chat_fn=boom)
    assert r["error"] == "api_error:RuntimeError"     # 只有类名
    assert secret not in r["error"]                    # 绝不含 key


def test_rerank_corrupt_cache_treated_as_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    cands = _cands3()
    tmp_path.mkdir(exist_ok=True)
    # 预置一个损坏的缓存文件（非法 JSON）
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text("{ broken json", encoding="utf-8")

    calls = {"n": 0}

    def good(system, user):
        calls["n"] += 1
        return ('{"ranked_unit_ids": ["a->b","c->d","e->f"]}', 5, 3, 0.01)
    r = rerank.rerank_units("q", cands, use_cache=True, chat_fn=good)
    assert calls["n"] == 1                             # 坏缓存 → 重新调用
    assert r["parse_ok"] and r["order"] == ["a->b", "c->d", "e->f"]
    assert r["cache_error"] and r["cache_error"].startswith("cache_error:")
    assert r["cached"] is False


def test_rerank_cache_missing_field_treated_as_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    cands = _cands3()
    tmp_path.mkdir(exist_ok=True)
    # 合法 JSON 但缺 telemetry 字段
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text(
        '{"order": ["a->b","c->d","e->f"], "parse_ok": true, "error": null}', encoding="utf-8")

    def good(system, user):
        return ('{"ranked_unit_ids": ["a->b","c->d","e->f"]}', 5, 3, 0.01)
    r = rerank.rerank_units("q", cands, use_cache=True, chat_fn=good)
    assert r["cache_error"] == "cache_error:missing_field"   # 字段缺失被识别（具体错误码）
    assert r["cached"] is False and r["parse_ok"]


def test_rerank_valid_cache_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    cands = _cands3()

    def good(system, user):
        return ('{"ranked_unit_ids": ["a->b","c->d","e->f"]}', 5, 3, 0.01)
    r1 = rerank.rerank_units("q", cands, use_cache=True, chat_fn=good)
    assert r1["cached"] is False
    # 第二次应命中缓存，不再调用（传入会抛错的 chat_fn 证明未被调用）
    def boom(system, user):
        raise AssertionError("should not be called")
    r2 = rerank.rerank_units("q", cands, use_cache=True, chat_fn=boom)
    assert r2["cached"] is True and r2["order"] == ["a->b", "c->d", "e->f"]


# ── allow_api 硬闸门（离线，绝不触发真实 API）──────────────────────
def _valid_cache_rec(cands):
    order = [c["unit_id"] for c in cands]
    return {"question": "q", "unit_ids": order, "raw": "{}", "order": order,
            "parse_ok": True, "error": None,
            "telemetry": {"in_tokens": 1, "out_tokens": 1, "latency": 0.0, "api_calls": 1}}


def test_gate_cache_miss_default_no_api(tmp_path, monkeypatch):
    # 默认 allow_api=False + 缓存缺失 → api_disabled，绝不调用真实 API
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    # 用会抛错的 _call 证明真实路径未被触发
    monkeypatch.setattr(rerank, "_call", lambda s, u: (_ for _ in ()).throw(AssertionError("API called!")))
    r = rerank.rerank_units("q", _cands3(), use_cache=True)   # chat_fn=None, allow_api=False
    assert r["order"] is None and r["error"] == "api_disabled"
    assert r["telemetry"]["api_calls"] == 0
    assert r["cache_error"] == "cache_miss"


def test_gate_corrupt_cache_default_no_api(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "_call", lambda s, u: (_ for _ in ()).throw(AssertionError("API called!")))
    cands = _cands3()
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text("{ broken", encoding="utf-8")
    r = rerank.rerank_units("q", cands, use_cache=True)       # 默认闸门关闭
    assert r["order"] is None and r["error"] == "api_disabled"
    assert r["telemetry"]["api_calls"] == 0
    assert r["cache_error"].startswith("cache_error:")        # 记录损坏原因


def test_gate_valid_cache_read_regardless_of_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "_call", lambda s, u: (_ for _ in ()).throw(AssertionError("API called!")))
    cands = _cands3()
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text(
        json.dumps(_valid_cache_rec(cands)), encoding="utf-8")
    r = rerank.rerank_units("q", cands, use_cache=True)       # allow_api=False 仍可读有效缓存
    assert r["cached"] is True and r["parse_ok"] and r["order"] == [c["unit_id"] for c in cands]
    assert r["telemetry"]["api_calls"] == 1                   # 来自缓存记录，非本次调用


def test_gate_allow_api_true_permits_real_call(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    called = {"n": 0}

    def fake_call(system, user):
        called["n"] += 1
        cands = _cands3()
        return ('{"ranked_unit_ids": ["a->b","c->d","e->f"]}', 3, 2, 0.01)
    monkeypatch.setattr(rerank, "_call", fake_call)
    r = rerank.rerank_units("q", _cands3(), use_cache=False, allow_api=True)  # 显式放行
    assert called["n"] == 1 and r["parse_ok"] and r["order"] == ["a->b", "c->d", "e->f"]


def test_gate_stub_chat_fn_bypasses_switch(tmp_path, monkeypatch):
    # 测试桩 chat_fn 不受 allow_api 闸门限制（即便 allow_api=False）
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    stub = lambda s, u: ('{"ranked_unit_ids": ["a->b","c->d","e->f"]}', 1, 1, 0.0)  # noqa: E731
    r = rerank.rerank_units("q", _cands3(), use_cache=False, chat_fn=stub, allow_api=False)
    assert r["parse_ok"] and r["order"] == ["a->b", "c->d", "e->f"]


# ── Codex 审查回归测试 ──────────────────────────────────────────────
def _cand(uid, anchor_rank=0, delta=1, atext="atext", ntext="ntext"):
    return {"unit_id": uid, "anchor_text": atext, "neighbor_text": ntext,
            "delta": delta, "anchor_rank": anchor_rank}


def test_import_and_cache_read_need_no_key(tmp_path, monkeypatch):
    # 无 API key 也能 import 且读取有效缓存：client 延迟创建，缓存命中不实例化
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "_client", None)
    monkeypatch.setattr(rerank, "_get_client",
                        lambda: (_ for _ in ()).throw(AssertionError("client created!")))
    cands = _cands3()
    order = [c["unit_id"] for c in cands]
    rec = {"question": "q", "unit_ids": order, "raw": "{}", "order": order,
           "parse_ok": True, "error": None,
           "telemetry": {"in_tokens": 1, "out_tokens": 1, "latency": 0.0, "api_calls": 1}}
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text(json.dumps(rec), encoding="utf-8")
    r = rerank.rerank_units("q", cands, use_cache=True)   # 缓存命中，不需 key/不创建 client
    assert r["cached"] and r["order"] == order


def test_validate_no_crash_on_unhashable_types():
    cands = _cands3()
    # order 里含 list / dict → 不能崩溃，应判 non_string_id
    order, err = rerank.validate('{"ranked_unit_ids": [["a"], {"x": 1}, "e->f"]}', cands)
    assert order is None and err == "non_string_id"


def test_semantically_invalid_cache_rejected(tmp_path, monkeypatch):
    # 缓存 order 是合法 JSON，但与当前候选 unit_id 不匹配（缺失/多余）→ 按损坏处理
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "_call",
                        lambda s, u: (_ for _ in ()).throw(AssertionError("API called!")))
    cands = _cands3()
    bad = {"question": "q", "unit_ids": [], "raw": "{}",
           "order": ["a->b", "c->d"],       # 缺 e->f
           "parse_ok": True, "error": None,
           "telemetry": {"in_tokens": 1, "out_tokens": 1, "latency": 0.0, "api_calls": 1}}
    (tmp_path / f"{rerank.cache_key('q', cands)}.json").write_text(json.dumps(bad), encoding="utf-8")
    r = rerank.rerank_units("q", cands, use_cache=True)   # allow_api=False
    assert r["order"] is None and r["error"] == "api_disabled"
    assert r["telemetry"]["api_calls"] == 0
    assert r["cache_error"] == "cache_error:count_mismatch"


def test_cache_key_changes_when_prompt_fields_change():
    base = [_cand("a->b", anchor_rank=0, ntext="N1")]
    k0 = rerank.cache_key("q", base)
    # 改 anchor_rank
    assert rerank.cache_key("q", [_cand("a->b", anchor_rank=5, ntext="N1")]) != k0
    # 改 neighbor 文本
    assert rerank.cache_key("q", [_cand("a->b", anchor_rank=0, ntext="N2")]) != k0
    # 改 direction/delta
    assert rerank.cache_key("q", [_cand("a->b", anchor_rank=0, delta=-1, ntext="N1")]) != k0
    # 改 question
    assert rerank.cache_key("q2", base) != k0
    # 全同 → 同键
    assert rerank.cache_key("q", [_cand("a->b", anchor_rank=0, ntext="N1")]) == k0


def test_cache_only_never_calls_real_api(tmp_path, monkeypatch):
    # cache-only + 缓存缺失：绝不触发 _call
    monkeypatch.setattr(rerank, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "_call",
                        lambda s, u: (_ for _ in ()).throw(AssertionError("real API called!")))
    r = rerank.rerank_units("q", _cands3(), use_cache=True)   # allow_api=False, chat_fn=None
    assert r["error"] == "api_disabled" and r["telemetry"]["api_calls"] == 0


def test_build_candidates_unknown_mode_raises():
    import pytest as _pt
    with _pt.raises(ValueError):
        build_candidates([_ch("a", 0)], [{"id": "a#0", "source": "a", "chunk_index": 0}],
                         {("a", 0): _ch("a", 0)}, "N-bogus")


def test_incomplete_gate_rejects_errors():
    from scripts.exp_rerank_dev import completeness_checks
    # 全绿 → 完整
    ok = completeness_checks(24, cached=24, parse_fail=0, cache_err=0,
                             api_err=0, api_disabled=0, this_run_calls=0)
    assert all(ok.values())
    # 任一错误都不得判完整（不会误报 PASS）
    assert not all(completeness_checks(24, 23, 0, 0, 0, 0, 0).values())   # 缓存不足
    assert not all(completeness_checks(24, 24, 1, 0, 0, 0, 0).values())   # parse_fail
    assert not all(completeness_checks(24, 24, 0, 1, 0, 0, 0).values())   # cache_error
    assert not all(completeness_checks(24, 24, 0, 0, 1, 0, 0).values())   # api_error
    assert not all(completeness_checks(24, 24, 0, 0, 0, 1, 0).values())   # api_disabled
    assert not all(completeness_checks(24, 24, 0, 0, 0, 0, 1).values())   # 本次真实调用


# ── 退出码回归（Fix 1，离线）────────────────────────────────────────
def test_result_code_mapping():
    from scripts.exp_rerank_dev import _result_code, completeness_checks
    complete = completeness_checks(24, 24, 0, 0, 0, 0, 0)
    assert all(complete.values())
    assert _result_code(complete, True) == 0     # 完整 + PASS
    assert _result_code(complete, False) == 1     # 完整 + 指标 FAIL
    incomplete = completeness_checks(24, 23, 0, 0, 0, 0, 0)
    assert _result_code(incomplete, True) == 2    # 不完整 → 2（无视 passed）
    assert _result_code(incomplete, None) == 2
    # 任一错误项都归为不完整 → 2
    for bad in (completeness_checks(24, 24, 1, 0, 0, 0, 0),
                completeness_checks(24, 24, 0, 1, 0, 0, 0),
                completeness_checks(24, 24, 0, 0, 1, 0, 0),
                completeness_checks(24, 24, 0, 0, 0, 1, 0),
                completeness_checks(24, 24, 0, 0, 0, 0, 1)):
        assert _result_code(bad, True) == 2


def test_main_allow_api_invalid_key_returns_2(monkeypatch):
    import sys
    import scripts.exp_rerank_dev as exp
    # key 无效 + --allow-api → 返回 2，且在触及索引/Ollama 之前就退出
    monkeypatch.setattr(exp, "_key_ok", lambda: False)
    monkeypatch.setattr(sys, "argv", ["exp_rerank_dev", "--allow-api"])
    assert exp.main() == 2


# ── 未知 fusion / method 提前抛错回归（离线）────────────────────────
def test_run_unknown_fusion_raises_before_retrieval(monkeypatch):
    from app import retrieval_pipeline as rp
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return []
    monkeypatch.setattr(rp.hybrid, "hybrid_search", _spy)   # 若被调用会计数
    with pytest.raises(ValueError):
        rp.run("q", k_final=6, w=1, fusion="A-typo")
    assert called["n"] == 0                                 # 校验先于检索，hybrid_search 未被调用


def test_score_windows_unknown_method_raises_without_mutation():
    win = {"anchor_source": "a", "anchor_idx": 0, "anchor_rank": 0,
           "chunks": [{"id": "a#0"}], "chunk_ids": ["a#0"]}
    windows = [win]
    with pytest.raises(ValueError):
        score_windows(windows, [1.0], {"a#0": [1.0]}, "WS-typo")
    assert "score" not in win          # 传入 windows 未被修改（无新增 score 字段）
    assert windows == [win]            # 列表内容不变

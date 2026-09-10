"""dev-only A/B 最终答案盲测工具的离线测试（不触网、不调用 DeepSeek、不连 Chroma）。"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()   # 防御：即便有惰性导入也不碰真实索引

from scripts import eval_generation_ab_dev as gab  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _hit(source, idx, text=None):
    return {"source": source, "chunk_index": idx, "text": text or f"{source} chunk {idx} text"}


def _fake_retrieve(n=6):
    def r(question):
        return [_hit("tutorial/x.md", i) for i in range(n)]
    return r


def _write_cache(out_dir, key, cid, k, answer="ans", in_tok=100, out_tok=20, lat=1.0):
    p = gab._cache_dir(out_dir) / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "id": cid, "k": k, "cache_key": key,
        "model": gab.EVAL_MODEL, "thinking": gab.EVAL_THINKING,
        "max_tokens": gab.EVAL_MAX_TOKENS, "temperature": gab.EVAL_TEMPERATURE,
        "prompt_version": gab.PROMPT_VERSION, "answer": answer,
        "telemetry": {"in_tokens": in_tok, "out_tokens": out_tok, "latency": lat},
    }, ensure_ascii=False), encoding="utf-8")


def _fingerprint(paths):
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes() if p.exists() else b"<none>")
    return h.hexdigest()


# ── 基础：24/48、前缀、无 test 引用 ────────────────────────────────
def test_loads_24_dev_and_48_plan_items():
    dev = gab.load_dev()
    assert len(dev) == 24
    plan = gab.build_plan(dev, _fake_retrieve(6))
    assert len(plan) == 48 and len({p["id"] for p in plan}) == 24


def test_no_reference_to_test_split():
    src = (REPO / "scripts" / "eval_generation_ab_dev.py").read_text(encoding="utf-8")
    assert "test.jsonl" not in src and "--split" not in src


def test_k4_is_strict_prefix_of_k6():
    plan = gab.build_plan([{"id": "d1", "question": "q1"}], _fake_retrieve(8))
    by = {}
    for p in plan:
        by.setdefault(p["id"], {})[p["k"]] = p["source_ids"]
    for ks in by.values():
        assert len(ks[6]) == 6 and ks[4] == ks[6][:4]
    gab.assert_prefix(plan)


def test_dangerous_params_raise():
    dev = [{"id": "d1", "question": "q1"}]
    with pytest.raises(ValueError):
        gab.build_plan(dev, _fake_retrieve(6), k_small=6, k_big=4)
    with pytest.raises(ValueError):
        gab.build_plan(dev, _fake_retrieve(3), k_big=6)


def test_cache_key_sensitivity():
    q, ctx = "question", "context-A"
    base = gab.cache_key(q, ctx)
    assert gab.cache_key(q, "context-B") != base
    assert gab.cache_key("other-q", ctx) != base
    assert gab.cache_key(q, ctx, model="other-model") != base
    assert gab.cache_key(q, ctx, thinking="enabled") != base
    assert gab.cache_key(q, ctx, max_tokens=512) != base
    assert gab.cache_key(q, ctx, prompt_version="genab-v3") != base
    assert gab.cache_key(q, ctx, system="different") != base
    assert gab.cache_key(q, ctx, temperature=0.7) != base
    assert gab.cache_key(q, ctx) == base


def test_experiment_id_binds_generation_config(monkeypatch):
    dev = [{"id": "d1", "question": "q1"}]
    plan = gab.build_plan(dev, _fake_retrieve(6))
    mapping = {"d1": {"A": "k4", "B": "k6"}}
    base = gab.compute_experiment_id(dev, plan, mapping, gab.AB_SEED)

    monkeypatch.setattr(gab, "EVAL_THINKING", "enabled")
    assert gab.compute_experiment_id(dev, plan, mapping, gab.AB_SEED) != base
    monkeypatch.setattr(gab, "EVAL_THINKING", "disabled")
    monkeypatch.setattr(gab, "EVAL_MAX_TOKENS", 512)
    assert gab.compute_experiment_id(dev, plan, mapping, gab.AB_SEED) != base


def test_build_citation_evidence_resolved_missing_and_unresolved():
    refs = {"tutorial/a.md#0": "support A", "tutorial/b.md#2": "support B"}
    resolved = gab.build_citation_evidence(
        "answer [来源: tutorial/a.md#0, tutorial/b.md#2]", refs)
    assert resolved == {"status": "resolved", "items": [
        {"citation": "tutorial/a.md#0", "status": "resolved", "text": "support A"},
        {"citation": "tutorial/b.md#2", "status": "resolved", "text": "support B"},
    ]}
    assert gab.build_citation_evidence("answer without citation", refs) == {
        "status": "missing", "items": []}
    unresolved = gab.build_citation_evidence(
        "answer [来源：tutorial/missing.md#9] [来源: malformed]", refs)
    assert unresolved["status"] == "unresolved"
    assert [item["status"] for item in unresolved["items"]] == ["not_found", "invalid_format"]


# ── A/B 平衡 12/12 ─────────────────────────────────────────────────
def test_ab_mapping_balanced_and_reproducible():
    ids = [f"d{i:02d}" for i in range(24)]
    m1 = gab.make_ab_mapping(ids, seed=20260823)
    assert m1 == gab.make_ab_mapping(ids, seed=20260823)
    assert sum(v["A"] == "k4" for v in m1.values()) == 12
    assert sum(v["A"] == "k6" for v in m1.values()) == 12


# ── telemetry / 缓存加固 ──────────────────────────────────────────
def test_generate_answer_returns_telemetry_tuple():
    ans, i, o, lat = gab.generate_answer("s", "u", chat_fn=lambda s, u: ("A", 12, 3, 0.4))
    assert (ans, i, o, lat) == ("A", 12, 3, 0.4)


def test_generation_config_is_eval_only_and_explicit():
    assert gab.EVAL_MODEL == "deepseek-v4-flash"
    assert gab.EVAL_THINKING == "disabled"
    assert gab.EVAL_MAX_TOKENS == 1024


def test_generate_answer_sends_frozen_v4_request(monkeypatch):
    seen = {}

    class Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            usage = type("Usage", (), {"prompt_tokens": 12, "completion_tokens": 3})()
            message = type("Message", (), {"content": "A"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"usage": usage, "choices": [choice]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(gab, "_get_client", lambda: client)
    answer = gab.generate_answer("system", "user")

    assert answer[:3] == ("A", 12, 3)
    assert seen["model"] == gab.EVAL_MODEL
    assert seen["temperature"] == gab.EVAL_TEMPERATURE
    assert seen["max_tokens"] == gab.EVAL_MAX_TOKENS
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}}


def test_telemetry_validation_rules():
    ok = {"in_tokens": 10, "out_tokens": 2, "latency": 0.5}
    assert gab._valid_telemetry(ok)
    assert not gab._valid_telemetry({"in_tokens": True, "out_tokens": 2, "latency": 0.5})   # bool
    assert not gab._valid_telemetry({"in_tokens": 1.5, "out_tokens": 2, "latency": 0.5})    # float token
    assert not gab._valid_telemetry({"in_tokens": -1, "out_tokens": 2, "latency": 0.5})     # 负
    assert not gab._valid_telemetry({"in_tokens": 1, "out_tokens": 2, "latency": -0.1})     # 负延迟
    assert not gab._valid_telemetry({"in_tokens": 1, "out_tokens": 2, "latency": float("inf")})  # Inf
    assert not gab._valid_telemetry({"in_tokens": 1, "out_tokens": 2, "latency": float("nan")})  # NaN
    assert not gab._valid_telemetry({"in_tokens": 1, "out_tokens": 2})                       # 缺字段


def test_cache_record_revalidation(tmp_path):
    _write_cache(tmp_path, "k1", "d1", 4)
    assert gab.load_answer(tmp_path, "k1", "d1", 4) == "ans"
    assert gab.load_answer(tmp_path, "k1", "OTHER", 4) is None      # id 不符
    assert gab.load_answer(tmp_path, "k1", "d1", 6) is None         # k 不符
    # 空白 answer → miss
    _write_cache(tmp_path, "k2", "d1", 4, answer="   ")
    assert gab.load_answer(tmp_path, "k2", "d1", 4) is None
    # generation config 任一不符 → miss
    p = gab._cache_dir(tmp_path) / "k1.json"
    rec = json.loads(p.read_text())
    rec["thinking"] = "enabled"
    p.write_text(json.dumps(rec), encoding="utf-8")
    assert gab.load_answer(tmp_path, "k1", "d1", 4) is None


# ── 评分校验 ───────────────────────────────────────────────────────
def _good_scores(ids):
    return [{"id": i, "a_correctness": 2, "b_correctness": 1,
             "a_citation_correct": 1, "b_citation_correct": 0,
             "reviewer_notes": ""} for i in ids]


def _good_faith_scores(ids):
    return [{"id": i, "a_faithfulness": 1, "b_faithfulness": 0,
             "reviewer_notes": ""} for i in ids]


def test_validate_scores():
    ids = ["d0", "d1"]
    assert gab.validate_scores(_good_scores(ids), ids) == []
    bad = _good_scores(ids); bad[0]["a_correctness"] = True
    assert any("a_correctness" in e for e in gab.validate_scores(bad, ids))
    bad = _good_scores(ids); bad[0]["b_correctness"] = 3
    assert any("b_correctness" in e for e in gab.validate_scores(bad, ids))
    bad = _good_scores(ids); bad[0]["b_citation_correct"] = -1
    assert any("b_citation_correct" in e for e in gab.validate_scores(bad, ids))
    bad = _good_scores(ids); del bad[0]["a_correctness"]
    assert any("a_correctness" in e for e in gab.validate_scores(bad, ids))
    bad = _good_scores(ids); bad[1]["id"] = "d0"
    assert any("重复" in e for e in gab.validate_scores(bad, ids))
    assert any("缺题" in e for e in gab.validate_scores(_good_scores(["d0"]), ids))
    assert gab.validate_faithfulness_scores(_good_faith_scores(ids), ids) == []
    bad_faith = _good_faith_scores(ids); bad_faith[0]["a_faithfulness"] = "1"
    assert any("a_faithfulness" in e
               for e in gab.validate_faithfulness_scores(bad_faith, ids))


# ── mapping / packet 校验 ─────────────────────────────────────────
def test_validate_mapping():
    ids = [f"d{i}" for i in range(24)]
    assert gab.validate_mapping(gab.make_ab_mapping(ids), ids) == []
    m = gab.make_ab_mapping(ids); m["d0"] = {"A": "k4", "B": "k4"}          # 非法标签
    assert gab.validate_mapping(m, ids)
    m = gab.make_ab_mapping(ids); del m["d0"]                               # 缺题
    assert gab.validate_mapping(m, ids)
    # A 位置不平衡（全 k4）
    unbal = {i: {"A": "k4", "B": "k6"} for i in ids}
    assert any("平衡" in e for e in gab.validate_mapping(unbal, ids))


def _good_packet(ids):
    evidence = {"status": "resolved", "items": [
        {"citation": "s.md#0", "status": "resolved", "text": "support"}]}
    return [{"id": i, "question": "q", "expected_source": "s.md", "evidence_quote": "ev",
             "answer_a": "aa", "answer_b": "bb",
             "citation_evidence_a": dict(evidence),
             "citation_evidence_b": dict(evidence)} for i in ids]


def test_validate_packet():
    dev = [{"id": "d0"}, {"id": "d1"}]
    ids = ["d0", "d1"]
    assert gab.validate_packet(_good_packet(ids), dev) == []
    bad = _good_packet(ids); bad[0]["answer_a"] = "   "
    assert any("answer_a" in e for e in gab.validate_packet(bad, dev))
    bad = _good_packet(ids); bad[0]["answer_b"] = 123
    assert any("answer_b" in e for e in gab.validate_packet(bad, dev))
    bad = _good_packet(ids); bad[0]["citation_evidence_a"] = {"status": "resolved", "items": []}
    assert any("citation_evidence_a" in e for e in gab.validate_packet(bad, dev))
    bad = _good_packet(ids); bad[0]["reference_source_ids"] = []
    assert any("泄露" in e for e in gab.validate_packet(bad, dev))
    bad = _good_packet(ids); bad[0]["telemetry_a"] = {"in_tokens": 1}
    assert any("泄露" in e for e in gab.validate_packet(bad, dev))
    bad = _good_packet(ids); bad[1]["id"] = "d0"
    assert any("重复" in e for e in gab.validate_packet(bad, dev))


# ── 输出目录约束 ───────────────────────────────────────────────────
def test_output_dir_under_eval_outputs():
    assert str(gab.OUTPUT_DIR).replace(os.sep, "/").startswith("eval/outputs/")


# ── prepare dry-run：无 client/无调用/退出 2/产物含 experiment_id ────
def test_prepare_dryrun_no_client_and_wrapped(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    monkeypatch.setattr(gab, "OUTPUT_DIR", out)
    monkeypatch.setattr(gab, "_default_retrieve", _fake_retrieve(6))
    monkeypatch.setattr(gab, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("client!")))
    monkeypatch.setattr(gab, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gen!")))
    monkeypatch.setattr(sys, "argv", ["p"])
    assert gab.main() == 2
    for f in (gab.PLAN_FILE, gab.MAPPING_FILE, gab.PACKET_FILE, gab.SCORES_FILE, gab.STATUS_FILE):
        assert (out / f).exists()
    assert not (out / "cache").exists()
    eids = {json.loads((out / f).read_text())["experiment_id"]
            for f in (gab.PLAN_FILE, gab.MAPPING_FILE, gab.PACKET_FILE, gab.SCORES_FILE)}
    assert len(eids) == 1 and None not in eids
    status = json.loads((out / gab.STATUS_FILE).read_text())
    assert status["api_attempts"] == 0 and status["api_successes"] == 0 and status["answers_ready"] == 0
    assert status["generation_config"] == {
        "model": gab.EVAL_MODEL, "thinking": gab.EVAL_THINKING,
        "max_tokens": gab.EVAL_MAX_TOKENS, "temperature": gab.EVAL_TEMPERATURE,
    }
    packet = json.loads((out / gab.PACKET_FILE).read_text())["packet"]
    forbidden = {"source_ids_a", "source_ids_b", "context_a", "context_b",
                 "telemetry_a", "telemetry_b", "reference_source_ids", "reference_context"}
    assert all(not (forbidden & set(row)) for row in packet)
    assert all("citation_evidence_a" in row and "citation_evidence_b" in row for row in packet)
    assert not (out / gab.FAITH_PACKET_FILE).exists()
    assert not (out / gab.FAITH_SCORES_FILE).exists()


# ── 错配防护：已填评分后换 seed 再 prepare → 4 产物不变且退出非零 ──
def test_prepare_guard_preserves_artifacts_on_seed_change(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    monkeypatch.setattr(gab, "OUTPUT_DIR", out)
    monkeypatch.setattr(gab, "_default_retrieve", _fake_retrieve(6))
    monkeypatch.setattr(gab, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("client!")))
    # 首次 prepare（种子默认）
    monkeypatch.setattr(sys, "argv", ["p"])
    gab.main()
    arts = [out / f for f in (gab.PLAN_FILE, gab.MAPPING_FILE, gab.PACKET_FILE, gab.SCORES_FILE)]
    # 人工填写评分（使 scores 非空白脚手架）
    sc = json.loads((out / gab.SCORES_FILE).read_text())
    sc["scores"][0]["a_correctness"] = 2
    (out / gab.SCORES_FILE).write_text(json.dumps(sc, ensure_ascii=False), encoding="utf-8")
    fp_before = _fingerprint(arts)
    # 换 seed 再 prepare，且不加 --force → 必须拒绝、非零、4 产物不变
    monkeypatch.setattr(gab, "_default_retrieve",
                        lambda: (_ for _ in ()).throw(AssertionError("retrieval must NOT run!")))
    monkeypatch.setattr(sys, "argv", ["p", "--seed", "999"])
    rc = gab.main()
    assert rc == 2
    assert _fingerprint(arts) == fp_before                 # plan/mapping/packet/scores 内容不变


# ── 空 key 可导入运行（子进程）─────────────────────────────────────
def test_empty_key_import_in_subprocess():
    env = dict(os.environ); env["DEEPSEEK_API_KEY"] = ""; env["CHROMA_DIR"] = tempfile.mkdtemp()
    code = ("import scripts.eval_generation_ab_dev as m; assert m._client is None; print('OK', m.PROMPT_VERSION)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ── summarize：完整流程 → 指标正确；experiment_id 一致 ──────────────
def _full_prepare(out, monkeypatch):
    monkeypatch.setattr(gab, "OUTPUT_DIR", out)
    monkeypatch.setattr(gab, "_default_retrieve", _fake_retrieve(6))
    monkeypatch.setattr(gab, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("client!")))
    dev = gab.load_dev()
    plan = gab.build_plan(dev, _fake_retrieve(6))
    for p in plan:
        _write_cache(out, p["cache_key"], p["id"], p["k"], answer=f"ans-k{p['k']}",
                     in_tok=100 + p["k"], out_tok=20 + p["k"], lat=1.0 + 0.1 * p["k"])
    monkeypatch.setattr(sys, "argv", ["p"])
    rc = gab.main()
    return dev, rc


def _fill_primary_scores(out, dev, mapping=None, k6_only=False):
    rows = []
    for it in dev:
        row = {"id": it["id"], "reviewer_notes": ""}
        for side in ("a", "b"):
            good = (not k6_only or
                    (mapping[it["id"]][side.upper()] == "k6"))
            row[f"{side}_correctness"] = 2 if good else 0
            row[f"{side}_citation_correct"] = 1
        rows.append(row)
    obj = json.loads((out / gab.SCORES_FILE).read_text())
    obj["scores"] = rows
    (out / gab.SCORES_FILE).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _prepare_faithfulness(out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["p", "--prepare-faithfulness"])
    return gab.main()


def _fill_faith_scores(out, dev, value=1):
    obj = json.loads((out / gab.FAITH_SCORES_FILE).read_text())
    obj["scores"] = [{"id": it["id"], "a_faithfulness": value,
                      "b_faithfulness": value, "reviewer_notes": ""} for it in dev]
    (out / gab.FAITH_SCORES_FILE).write_text(
        json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_prepare_ready_then_summarize(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, rc = _full_prepare(out, monkeypatch)
    assert rc == 0
    mapping = json.loads((out / gab.MAPPING_FILE).read_text())["mapping"]
    _fill_primary_scores(out, dev, mapping, k6_only=True)
    assert _prepare_faithfulness(out, monkeypatch) == 0
    _fill_faith_scores(out, dev)
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 0
    summary = json.loads((out / gab.SUMMARY_FILE).read_text())
    assert summary["quality"]["k6"]["overall_pass"] == 24
    assert summary["quality"]["k4"]["overall_pass"] == 0
    assert summary["paired"]["net_k6_wins"] == 24
    assert summary["decision_gate"]["k6_candidate"] is True
    assert summary["cost"]["k6"]["avg_in_tokens"] == 106 and summary["cost"]["k4"]["avg_in_tokens"] == 104
    assert summary["cost"]["k6"]["n"] == 24 and "p95_latency" in summary["cost"]["k6"]
    assert summary["experiment_id"] == json.loads((out / gab.PLAN_FILE).read_text())["experiment_id"]


def test_faithfulness_stage_exposes_context_only_after_primary_lock(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    primary = json.loads((out / gab.PACKET_FILE).read_text())["packet"]
    assert all("context_a" not in row and "telemetry_a" not in row for row in primary)

    _fill_primary_scores(out, dev)
    assert _prepare_faithfulness(out, monkeypatch) == 0
    faith = json.loads((out / gab.FAITH_PACKET_FILE).read_text())["packet"]
    assert all({len(row["source_ids_a"]), len(row["source_ids_b"])} == {4, 6} for row in faith)
    assert all("telemetry_a" not in row and "telemetry_b" not in row for row in faith)


def test_summarize_rejects_primary_scores_changed_after_lock(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _fill_primary_scores(out, dev)
    assert _prepare_faithfulness(out, monkeypatch) == 0
    _fill_faith_scores(out, dev)
    primary = json.loads((out / gab.SCORES_FILE).read_text())
    primary["scores"][0]["a_correctness"] = 0
    (out / gab.SCORES_FILE).write_text(json.dumps(primary), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 2
    assert not (out / gab.SUMMARY_FILE).exists()


def test_summarize_incomplete_scores_no_summary(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    _full_prepare(out, monkeypatch)                        # 空白脚手架评分
    monkeypatch.setattr(sys, "argv", ["p", "--prepare-faithfulness"])
    assert gab.main() == 2
    assert not (out / gab.FAITH_PACKET_FILE).exists()
    assert not (out / gab.SUMMARY_FILE).exists()


def test_summarize_experiment_id_mismatch(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _complete_two_stage_scores(out, dev, monkeypatch)
    # 篡改 scores 的 experiment_id → 不一致
    sc = json.loads((out / gab.SCORES_FILE).read_text())
    sc["experiment_id"] = "TAMPERED"
    (out / gab.SCORES_FILE).write_text(json.dumps(sc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 2
    assert not (out / gab.SUMMARY_FILE).exists()


def test_summarize_missing_files_exit2(tmp_path, monkeypatch):
    monkeypatch.setattr(gab, "OUTPUT_DIR", tmp_path / "empty")
    monkeypatch.setattr(gab, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("client!")))
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 2


# ── CLI 冲突 ───────────────────────────────────────────────────────
def test_cli_conflict_summarize_with_allow_api(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["p", "--summarize", "--allow-api"])
    assert gab.main() == 2
    monkeypatch.setattr(sys, "argv", ["p", "--summarize", "--force"])
    assert gab.main() == 2
    monkeypatch.setattr(sys, "argv", ["p", "--prepare-faithfulness", "--allow-api"])
    assert gab.main() == 2
    monkeypatch.setattr(sys, "argv", ["p", "--prepare-faithfulness", "--summarize"])
    assert gab.main() == 2


# ── rerank_cache 不变 ──────────────────────────────────────────────
def _rerank_fp():
    import hashlib
    d = REPO / "eval" / "rerank_cache"
    files = sorted(d.glob("*.json")) if d.exists() else []
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return len(files), h.hexdigest()


def test_rerank_cache_untouched(tmp_path, monkeypatch):
    before = _rerank_fp()
    monkeypatch.setattr(gab, "OUTPUT_DIR", tmp_path / "gen_ab")
    monkeypatch.setattr(gab, "_default_retrieve", _fake_retrieve(6))
    monkeypatch.setattr(gab, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("client!")))
    monkeypatch.setattr(sys, "argv", ["p"])
    gab.main()
    assert _rerank_fp() == before


# ── SYSTEM 与 rag 逐字一致 ─────────────────────────────────────────
def test_system_prompt_matches_rag_source():
    import ast
    tree = ast.parse((REPO / "app" / "rag.py").read_text(encoding="utf-8"))
    rag_system = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SYSTEM" for t in node.targets):
            rag_system = ast.literal_eval(node.value); break
    assert rag_system is not None and gab.SYSTEM == rag_system


# ── 收尾#1: 深度一致性校验 —— 篡改 plan / packet 派生字段必须失败 ────
def _complete_two_stage_scores(out, dev, monkeypatch):
    _fill_primary_scores(out, dev)
    assert _prepare_faithfulness(out, monkeypatch) == 0
    _fill_faith_scores(out, dev)


def test_summarize_rejects_tampered_plan(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _complete_two_stage_scores(out, dev, monkeypatch)
    plan_obj = json.loads((out / gab.PLAN_FILE).read_text())      # 保留 experiment_id 与 4 个 ID
    plan_obj["plan"][0]["context"] = plan_obj["plan"][0]["context"] + "TAMPER"  # 只改内容
    (out / gab.PLAN_FILE).write_text(json.dumps(plan_obj, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 2
    assert not (out / gab.SUMMARY_FILE).exists()


def test_summarize_rejects_tampered_packet(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _complete_two_stage_scores(out, dev, monkeypatch)
    pkt = json.loads((out / gab.PACKET_FILE).read_text())
    pkt["packet"][0]["citation_evidence_a"] = {
        "status": "resolved", "items": [
            {"citation": "WRONG#0", "status": "resolved", "text": "TAMPERED"}]}
    (out / gab.PACKET_FILE).write_text(json.dumps(pkt, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 2
    assert not (out / gab.SUMMARY_FILE).exists()


# ── 收尾#2: 旧 summary 失效 / guard 拒绝时保留 summary ──────────────
def test_force_rebuild_invalidates_old_summary(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _complete_two_stage_scores(out, dev, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 0 and (out / gab.SUMMARY_FILE).exists()
    # scores 已非空白脚手架；--force 重建整套 → 旧 summary 失效
    monkeypatch.setattr(sys, "argv", ["p", "--force"])
    gab.main()
    assert not (out / gab.SUMMARY_FILE).exists()


def test_guard_reject_keeps_old_summary(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    dev, _ = _full_prepare(out, monkeypatch)
    _complete_two_stage_scores(out, dev, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p", "--summarize"])
    assert gab.main() == 0 and (out / gab.SUMMARY_FILE).exists()
    # scores 非空白脚手架、无 --force → guard 拒绝，且不得动 summary
    monkeypatch.setattr(gab, "_default_retrieve",
                        lambda: (_ for _ in ()).throw(AssertionError("no retrieval on guard reject!")))
    monkeypatch.setattr(sys, "argv", ["p"])
    assert gab.main() == 2
    assert (out / gab.SUMMARY_FILE).exists()               # 原 summary 保留


# ── 收尾#3: 真实生成结果写缓存前严格校验 ───────────────────────────
def _prep_allow_api(out, monkeypatch, gen_fn):
    monkeypatch.setattr(gab, "OUTPUT_DIR", out)
    monkeypatch.setattr(gab, "_default_retrieve", _fake_retrieve(6))
    monkeypatch.setattr(gab, "_key_ok", lambda: True)
    monkeypatch.setattr(gab, "_get_client",
                        lambda: (_ for _ in ()).throw(AssertionError("real client!")))
    monkeypatch.setattr(gab, "generate_answer", gen_fn)


def test_generation_empty_answer_rejected(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    _prep_allow_api(out, monkeypatch, lambda s, u: ("   ", 10, 2, 0.5))   # strip 后空
    monkeypatch.setattr(sys, "argv", ["p", "--allow-api"])
    assert gab.main() == 2
    st = json.loads((out / gab.STATUS_FILE).read_text())
    assert st["error_type"] == "output_validation_error"
    assert st["api_errors"] >= 1 and st["api_successes"] == 0
    assert not (out / "cache").exists() or not any((out / "cache").iterdir())   # 未写缓存


def test_generation_bool_token_rejected(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    _prep_allow_api(out, monkeypatch, lambda s, u: ("ok", True, 2, 0.5))   # bool token
    monkeypatch.setattr(sys, "argv", ["p", "--allow-api"])
    assert gab.main() == 2
    st = json.loads((out / gab.STATUS_FILE).read_text())
    assert st["error_type"] == "output_validation_error" and st["api_successes"] == 0


def test_generation_nan_latency_rejected(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    _prep_allow_api(out, monkeypatch, lambda s, u: ("ok", 10, 2, float("nan")))
    monkeypatch.setattr(sys, "argv", ["p", "--allow-api"])
    assert gab.main() == 2
    assert json.loads((out / gab.STATUS_FILE).read_text())["error_type"] == "output_validation_error"


def test_generation_valid_writes_cache_and_succeeds(tmp_path, monkeypatch):
    out = tmp_path / "gen_ab"
    _prep_allow_api(out, monkeypatch, lambda s, u: ("a good grounded answer", 10, 3, 0.2))
    monkeypatch.setattr(sys, "argv", ["p", "--allow-api"])
    rc = gab.main()
    assert rc == 0
    st = json.loads((out / gab.STATUS_FILE).read_text())
    assert st["api_successes"] == 48 and st["api_errors"] == 0 and st["answers_ready"] == 48
    assert len(list((out / "cache").glob("*.json"))) == 48

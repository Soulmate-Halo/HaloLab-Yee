"""KAD core (yee.kad): the same policy logic as the vLLM adapter, and the controller behaviour."""
import hashlib
import json
import os
import random

from yee import kad
from vllm_d_recall.integrations import vllm_kad as vk

POLICIES = [
    kad.DEFAULT_POLICY,
    {"mode": "fixed", "fixed": 7},
    {"mode": "off"},
    {"mode": "ladder", "base": 3, "kmin": 2, "kmax": 9, "up_after": 2, "up_step": 3, "down": "step",
     "down_step": 1},
    {"mode": "ladder", "base": 20, "kmin": 14, "kmax": 12},
    {"mode": "LADDER", "down": " Accepted ", "hot_ttl": 0},
    {},
]
BAD = [None, [], "x", {"mode": 1}, {"mode": "nope"}, {"down": "x"}, {"base": True}, {"kmax": 1.5},
       {"down": 3}]


def _kad(**kw):
    kw.setdefault("enabled", True)
    kw.setdefault("policy_file", "")
    kw.setdefault("stats_file", "")
    kw.setdefault("log_s", 0)
    return kad.Kad(kw.pop("base", "t"), kw.pop("kmax_run", 12), **kw)


def test_parse_policy_same_as_vllm():
    for obj in POLICIES:
        assert kad.parse_policy(obj) == vk.parse_policy(obj), obj
    for obj in BAD:
        for fn in (kad.parse_policy, vk.parse_policy):
            try:
                fn(obj)
            except ValueError:
                continue
            raise AssertionError("accepted %r" % (obj,))
    assert kad.parse_policy(kad.DEFAULT_POLICY) == vk._default_policy()


def test_ladder_and_decide_same_as_vllm():
    rng = random.Random(641)
    for raw in POLICIES:
        pol = kad.parse_policy(raw)
        for kmax_run in (1, 5, 8, 12, 16):
            st_a = {"cur": pol["base"], "streak": 0, "hot_until": -1}
            st_b = dict(st_a)
            for step in range(1, 400):
                if rng.random() < 0.05:
                    st_a["hot_until"] = st_b["hot_until"] = step + pol["hot_ttl"]
                w = kad.decide_k(st_a, pol, step, kmax_run)
                assert w == vk.decide_w(st_b, pol, step, kmax_run)
                v = w if rng.random() < 0.8 else rng.randint(0, 13)
                a = rng.randint(0, max(0, v)) if rng.random() < 0.6 else v
                kad.ladder_update(st_a, v, a, pol)
                vk.ladder_update(st_b, v, a, pol)
                assert st_a == st_b
            assert kad.decide_k(None, pol, 1, kmax_run) == vk.decide_w(None, pol, 1, kmax_run)


def test_default_ladder_widths():
    k = _kad()
    seq = []
    for _ in range(6):
        k.begin_step()
        w = k.width("r")
        seq.append(w)
        k.observe("r", w, w)
    assert seq == [5, 7, 9, 11, 12, 12]
    k.begin_step()
    k.observe("r", 12, 3)
    assert k.width("r") == 5  # max(5, min(12, 3 + 1))
    k.observe("r", 5, 5)
    assert k.width("r") == 7
    k.observe("r", 7, 6)
    assert k.width("r") == 7  # max(5, min(7, 6 + 1))


def test_modes_caps_structured(tmp_path):
    p = tmp_path / "pol.json"
    k = _kad(policy_file=str(p), reload_s=0)
    assert k.width("a", structured=True) == 12
    p.write_text(json.dumps({"mode": "fixed", "fixed": 7}))
    k.begin_step()
    assert k.width("a") == 7
    k.set_kmax_run(5)
    assert k.width("a") == 5
    k.set_kmax_run(12)
    p.write_text(json.dumps({"mode": "off"}))
    k.begin_step()
    assert k.width("a") == 12


def test_disabled_is_inert(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"mode": "fixed", "fixed": 3}))
    s = tmp_path / "stats.json"
    k = _kad(enabled=False, policy_file=str(p), stats_file=str(s))
    for _ in range(3):
        k.begin_step()
        assert k.width("r") == 12
        assert k.batch_width(["r", "q"]) == 12
        k.observe("r", 12, 0)
        k.note_batch({"r": 12}, 12)
        k.mark_hot("r")
    k.maybe_log(force=True)
    assert k.policy() == kad.parse_policy(kad.DEFAULT_POLICY)
    assert not s.exists()
    snap = k.snapshot()
    assert snap["v_steps"] == 0 and snap["w_hist"] == {} and snap["tracked"] == 0


def test_reload_keeps_last_valid(tmp_path, capsys):
    p = tmp_path / "pol.json"
    k = _kad(policy_file=str(p), reload_s=0)
    assert k.policy()["mode"] == "ladder"  # no file: the default ladder
    data = json.dumps({"mode": "fixed", "fixed": 6}).encode()
    p.write_bytes(data)
    k.begin_step()
    assert k.width("r") == 6
    assert k.snapshot()["policy_sha256"] == hashlib.sha256(data).hexdigest()
    p.write_bytes(b"{not json")
    k.begin_step()
    assert k.width("r") == 6 and k.snapshot()["errors"] == 1
    p.write_bytes(json.dumps({"mode": "nope"}).encode())
    k.begin_step()
    assert k.width("r") == 6 and k.snapshot()["errors"] == 2
    os.remove(str(p))
    k.begin_step()
    assert k.width("r") == 6
    assert "[yee-kad] t policy loaded sha=%s" % hashlib.sha256(data).hexdigest()[:12] in capsys.readouterr().out


def test_reload_interval(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"mode": "fixed", "fixed": 6}))
    k = _kad(policy_file=str(p), reload_s=3600)
    assert k.width("r") == 6  # read when the controller is built
    p.write_text(json.dumps({"mode": "fixed", "fixed": 10, "pad": 1}))
    k.begin_step()
    assert k.width("r") == 6  # not looked at again inside RELOAD_S


def test_sync_ranks_switch_at_the_same_step(tmp_path):
    p0 = tmp_path / "pol0.json"
    p0.write_text(json.dumps({"mode": "fixed", "fixed": 8}))
    p1 = tmp_path / "pol1.json"  # a non-leader must never use its own file
    p1.write_text(json.dumps({"mode": "fixed", "fixed": 3}))
    box = {}

    def sync_for(leader):
        def sync(data):
            if leader:
                box["data"] = data
                return data
            return box.get("data")
        return sync

    r0 = _kad(policy_file=str(p0), leader=True, sync=sync_for(True), reload_every=4)
    r1 = _kad(policy_file=str(p1), leader=False, sync=sync_for(False), reload_every=4)
    for step in range(1, 14):
        if step == 6:
            p0.write_text(json.dumps({"mode": "fixed", "fixed": 10, "pad": 0}))
        r0.begin_step()  # rank 0 first: the box plays the broadcast
        r1.begin_step()
        assert r0.width("x") == r1.width("x")
        assert r0.width("x") == (8 if step < 8 else 10), step


def test_stats_and_log(tmp_path, capsys):
    s = tmp_path / "sub" / "stats.json"
    k = _kad(base="eng", stats_file=str(s))
    k.begin_step()
    ws = k.widths(["a", "b"], structured=[False, True])
    assert ws == {"a": 5, "b": 12}
    assert k.batch_width(["a", "b"], structured={"b"}) == 12
    assert k.batch_width([]) == 12
    k.note_batch(ws, k_used=12)
    k.observe("a", 12, 12)
    k.observe("b", 12, 4)
    k.maybe_log(force=True)
    snap = json.loads(s.read_text())
    assert snap["w_hist"] == {"5": 1, "12": 1} and snap["b_hist"] == {"12": 1}
    assert snap["mixed_steps"] == 1 and snap["acc_by_v"] == {"12": [2, 16]}
    assert snap["sum_v"] == 24 and snap["sum_a"] == 16 and snap["v_steps"] == 2
    assert snap["tracked"] == 2 and snap["kmax_run"] == 12 and snap["base"] == "eng"
    assert k.state("a") == {"cur": 7, "streak": 0, "hot_until": -1}
    assert "[yee-kad] eng steps=1 mixed=1 w_hist=5:1,12:1 b_hist=12:1" in capsys.readouterr().out
    k.release("a")
    k.retain(["zzz"])
    assert k.snapshot()["tracked"] == 0


def test_hot_window():
    k = _kad()
    k.begin_step()  # step 1
    k.mark_hot("r")  # hot until step 1 + 8
    assert k.width("r") == 12
    for _ in range(8):
        k.begin_step()  # step 9
    assert k.width("r") == 12
    k.begin_step()  # step 10
    assert k.width("r") == 5


def test_internal_error_falls_back(monkeypatch, capsys):
    k = _kad()

    def boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(kad, "decide_k", boom)
    for _ in range(7):
        assert k.width("r") == 12
    assert k.snapshot()["errors"] == 7
    out = capsys.readouterr().out
    assert out.count("width failed, the step keeps the native width") == 5


def test_warning_when_launch_k_is_not_12(capsys):
    _kad(kmax_run=8)
    assert ("[yee-kad] t WARNING launch K=8, KAD needs 12 (per-request width is capped at min(12, K))"
            in capsys.readouterr().out)
    _kad(kmax_run=12)
    _kad(kmax_run=8, enabled=False)
    assert "WARNING" not in capsys.readouterr().out


def test_environment(monkeypatch, tmp_path):
    for name in ("YEE_KAD_POLICY", "YEE_KAD_STATS_FILE", "YEE_KAD_LOG_S", "YEE_KAD_RELOAD_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YEE_KAD", "on")
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path))
    monkeypatch.setenv("YEE_KAD_RELOAD_EVERY", "0")
    k = kad.Kad("my eng", 12)
    assert k.enabled and kad.env_enabled()
    assert k.policy_file == os.path.join(str(tmp_path), "kad_policy.json")
    assert k.stats_file == os.path.join(str(tmp_path), "kad_stats_my_eng.json")
    assert k.reload_every == 1 and k.log_s == 30.0 and k.reload_s == 1.0
    monkeypatch.setenv("YEE_KAD", "0")
    assert not kad.Kad("x", 12).enabled

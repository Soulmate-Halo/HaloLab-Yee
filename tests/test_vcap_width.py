"""r703 选宽度器影子模式单测（随 yee/patches/mia-dsv41-vcap 发布）：与 3.7 对拍、选择与滞回、结算口径、
配置与异常隔离、verify_cap 联调、性能。"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PD = os.path.join(PKG, "yee", "patches", "mia-dsv41-vcap", "files")
FIX37 = os.path.join(HERE, "fixtures", "vcap_policy_37.py")  # Yee 3.7.0 原版 vcap_policy.py
sys.path.insert(0, PD)
import vcap_policy as vp  # noqa: E402

P7 = 7
W3 = {"widths": [4, 8, 13]}  # 默认只有 4 / 8；要测 13 的用例显式加上


def load_37():
    spec = importlib.util.spec_from_file_location("vcap_policy_37", FIX37)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drive(pol, gen, n, dt_ms, lag=1, t0=0.0):
    """模拟 overlap：第 s 步 prepare 之后结算第 s-lag 步。gen(s) -> (rids, k, acc, rec)。"""
    t = t0
    q = []
    for s in range(n):
        rids, k, acc, rec = gen(s)
        pol.prepare_step(rids, [0.0] * len(rids), t, need=False)
        q.append((rids, k, acc, rec, t))
        while len(q) > lag:
            r_, k_, a_, rc_, tp = q.pop(0)
            pol.observe(r_, k_, np.ones((len(r_), pol.P)), rc_, a_, tp)
        t += dt_ms / 1000.0
    return t


def churn_inputs(seed, n, P):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n):
        bs = int(rng.integers(1, 9))
        base = s // 25
        rids = ["r%d" % (base * 3 + j) for j in range(bs)]
        temps = [float(rng.choice([0.0, 0.7, 1.0])) for _ in range(bs)]
        cum = np.cumprod(rng.uniform(0.3, 1.0, size=(bs, P)), axis=1)
        k = rng.integers(1, P + 1, size=bs)
        acc = rng.integers(0, P + 1, size=bs)
        acc = np.where(rng.random(bs) < 0.9, np.minimum(acc, k), acc)  # 偶尔 acc > k
        rec = rng.random(bs) < 0.3
        out.append((rids, temps, (k, cum, rec, acc)))
    return out


# ---------------------------------------------------------------- 1. 与 3.7 逐位一致
@pytest.mark.parametrize("mode", ["legacy", "adapt"])
@pytest.mark.parametrize("width", ["off", "enabled", "mode"])
def test_identical_to_37(mode, width):
    v37 = load_37()
    cfg_old = {"mode": mode, "tag": "t"}
    cfg_new = dict(cfg_old)
    if width == "enabled":
        cfg_new["width"] = {"enabled": True}
    elif width == "mode":
        cfg_new["mode"] = "width"  # verify_cap 按 legacy 走；策略里开宽度影子
    old = v37.VcapPolicy(P7, cfg_old, seed=0)
    new = vp.VcapPolicy(P7, cfg_new, seed=0)
    assert (new._wd is not None) == (width != "off")
    need = mode == "adapt"
    pend = []
    t = 1000.0
    for s, (rids, temps, obs) in enumerate(churn_inputs(7, 300, P7)):
        t += 0.04 + 0.01 * (s % 3)
        o1 = old.prepare_step(rids, temps, t, need=need)
        o2 = new.prepare_step(rids, temps, t, need=need)
        if need:
            for x, y in zip(o1, o2):
                assert np.array_equal(x, y), s
        else:
            assert o1 is None and o2 is None
        pend.append((rids, obs, t))
        while len(pend) > 2:
            prids, (k, cum, rec, acc), tp = pend.pop(0)
            old.observe(prids, k, cum, rec, acc, tp)
            new.observe(prids, k, cum, rec, acc, tp)
        if s % 50 == 49:
            s1, s2 = old.stats(), new.stats()
            wv = s2.pop("width", None)
            assert (wv is not None) == (width != "off")
            assert "width_error" not in s2
            assert s1 == s2, s
            assert old.drain_finished() == new.drain_finished()
            if wv is not None:
                json.dumps(wv)
                assert wv["timed"] > 0 and wv["missing"] == 0
                assert all(isinstance(k_, str) for k_ in wv["by_bs"])


def test_widths_include_w0_and_cost_interp():
    assert vp.VcapPolicy(P7, {"mode": "width"})._wd.widths == [4, 8]
    assert vp.VcapPolicy(12, {"mode": "width"})._wd.widths == [4, 8, 13]
    wd = vp.VcapPolicy(5, {"mode": "width", "width": W3})._wd
    assert wd.W0 == 6 and wd.widths == [4, 6, 8, 13] and wd.nlo == 2
    live, dead = wd.coef(3)
    f = math.log2(3) - 1.0
    assert live == pytest.approx(2.2 + (1.67 - 2.2) * f) and dead == pytest.approx(0.63 + (0.55 - 0.63) * f)
    assert wd.coef(1) == (2.8, 0.58) and wd.coef(64) == (1.29, 0.52)


# ---------------------------------------------------------------- 2/3/4. 选择
def test_all_new_batch_picks_4():
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": W3})
    rng = np.random.default_rng(1)
    rids = ["n%d" % i for i in range(8)]

    def gen(s):
        k = rng.integers(1, 3, size=8)
        acc = np.minimum(rng.integers(0, 2, size=8), k)
        return rids, k, acc, np.zeros(8, dtype=bool)

    drive(pol, gen, 30, 95.0)
    pol.stats()
    drive(pol, gen, 60, 95.0, t0=10.0)
    w = pol.stats()["width"]
    b = w["by_bs"]["8"]
    assert b["n"] >= 50
    assert b["pick"]["4"] == b["n"] and b["pick"]["13"] == 0
    assert b["static"]["4"][0] == b["act"][0]  # acc <= 1：截到 3 个草稿不丢 token
    assert b["static"]["4"][0] / b["static"]["4"][1] > 1.10 * b["act"][0] / b["act"][1]
    assert b["sel"] == b["static"]["4"] and b["le0"] == b["static"]["4"]


@pytest.mark.parametrize("bs,dt,want", [(1, 45.0, 13), (8, 129.0, 8)])
def test_recall_saturated(bs, dt, want):
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": W3})
    rids = ["c%d" % i for i in range(bs)]
    full = np.full(bs, P7)

    def gen(s):
        return rids, full, full, np.ones(bs, dtype=bool)

    drive(pol, gen, 40, dt)
    pol.stats()
    drive(pol, gen, 40, dt, t0=10.0)
    b = pol.stats()["width"]["by_bs"][str(vp.bs_bucket(bs))]
    assert b["n"] > 30 and b["pick"][str(want)] == b["n"]
    assert b["pick_le0"]["8"] == b["n"]
    if want == 13:  # 越过块长的部分是估算：sel 高于 sel_lo，sel_lo 的 token 等于实际
        assert b["sel"][0] > b["sel_lo"][0] == b["act"][0]


def test_mixed_batch_matches_reference():
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": W3})
    rids = ["c0", "c1", "n0", "n1"]
    k = np.array([P7, P7, 2, 2])
    acc = np.array([P7, P7, 1, 1])
    rec = np.array([True, True, False, False])
    drive(pol, lambda s: (rids, k, acc, rec), 40, 60.0)
    wd = pol._wd
    qs = [pol._q_of(pol.reqs[r]) for r in rids]
    sc = wd.scores(rids, qs)

    # 参考实现：按规格从头算（39 个已结算步，每请求环满 16，全局环 156 行：一半召回满行、一半 new 行）
    def geo(r, m):
        return sum(r ** i for i in range(1, m + 1))

    ws, P, n, pn = [4, 8, 13], P7, 16, 2.0
    wr, wg = n / (n + pn), pn / (n + pn)
    E, L = [], []
    for W in ws:
        m = W - 1 - P
        e = l_ = 0.0
        for i in range(4):
            q = qs[i]
            full_a = min(P, W - 1) + (geo(q, m) if m > 0 else 0.0)
            full_k = min(P, W - 1) + max(m, 0)
            a_r, k_r = (full_a, full_k) if rec[i] else (min(1, W - 1), min(2, W - 1))
            a_g = (full_a + min(1, W - 1)) / 2.0
            k_g = (full_k + min(2, W - 1)) / 2.0
            e += 1 + wr * a_r + wg * a_g
            l_ += 1 + wr * k_r + wg * k_g
        E.append(e)
        L.append(l_)
    D = [4 * W - L[j] for j, W in enumerate(ws)]
    cl, cd = 1.67, 0.55
    T = [max(60.0 + cl * (L[j] - L[1]) + cd * (D[j] - D[1]), 15.0) for j in range(3)]
    S = [E[j] / T[j] for j in range(3)]
    assert sc["E"] == pytest.approx(E, rel=1e-9)
    assert sc["L"] == pytest.approx(L, rel=1e-9)
    assert sc["T"] == pytest.approx(T, rel=1e-9)
    best = int(np.argmax(S))
    assert ws[best] == 13 and S[best] > 1.03 * max(S[0], S[1])
    assert wd.widths[wd.cur] == 13 and wd.widths[wd.cur_le0] == 8


# ---------------------------------------------------------------- 5. 滞回
def test_pick_width_hysteresis():
    assert vp.pick_width([1.0, 1.02, 0.5], [0, 1, 2], 0, 0.03) == 0   # 只好 2%：不换
    assert vp.pick_width([1.0, 1.04, 0.5], [0, 1, 2], 0, 0.03) == 1   # 好 4%：换
    assert vp.pick_width([1.0, 0.9, 2.0], [0, 1], 2, 0.03) == 0       # 当前宽度不可选：换到可选里最好的
    assert vp.pick_width([0.5, 0.9, 0.7], [0, 1, 2], 1, 0.03) == 1


def test_policy_switch_counts():
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": {"switch": 10.0}})  # 门槛高到永远不换
    rng = np.random.default_rng(2)
    rids = ["n%d" % i for i in range(8)]
    drive(pol, lambda s: (rids, rng.integers(1, 3, size=8), np.zeros(8, dtype=int), np.zeros(8, dtype=bool)),
          30, 95.0)
    w = pol.stats()["width"]
    assert w["switches"] == 0 and w["by_bs"]["8"]["pick"]["8"] == w["by_bs"]["8"]["n"]


# ---------------------------------------------------------------- 6. 结算口径
def test_settlement_exact_and_untimed():
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": W3})
    rids = ["a", "b"]
    pol.prepare_step(rids, [0.0, 0.0], 0.0, need=False)
    pol.prepare_step(rids, [0.0, 0.0], 0.1, need=False)
    pol.observe(rids, np.array([3, 1]), np.ones((2, P7)), np.array([False, False]), np.array([2, 0]), 0.0)
    # t_prep=0.1 的下一步还没开头：挂起，等下一次 prepare 补步时
    pol.observe(rids, np.array([7, 7]), np.ones((2, P7)), np.array([True, False]), np.array([7, 7]), 0.1)
    w = pol.stats()["width"]
    assert w["steps"] == 2 and w["timed"] == 1 and w["untimed"] == 0 and w["missing"] == 0
    b = w["by_bs"]["2"]
    # A0 = 3 + 1，L0 = 4 + 2，D0 = 16 - 6；bs=2 的每行成本 live 2.2、dead 0.63
    assert b["n"] == 1 and b["act"] == [4.0, 100.0]
    assert b["static"]["4"] == [4.0, pytest.approx(100 + 0.63 * (2 - 10))]
    assert b["static"]["8"] == [4.0, 100.0]
    assert b["static"]["13"] == [4.0, pytest.approx(100 + 0.63 * (20 - 10))]
    assert b["pred"][1] == b["sel"][0]
    pol.prepare_step(rids, [0.0, 0.0], 5.0, need=False)  # 隔 4.9 秒：超过 MAX_STEP_GAP_MS
    w2 = pol.stats()["width"]
    assert w2["untimed"] == 1 and w2["timed"] == 0 and w2["by_bs"] == {}
    assert w2["t_w0"] == {"2": 100.0}


# ---------------------------------------------------------------- 7. 配置与异常隔离
def test_bad_width_config_reports_error_and_keeps_policy():
    pol = vp.VcapPolicy(P7, {"mode": "width", "width": {"widths": "4,8"}})
    assert pol._wd is None
    assert pol.prepare_step(["a"], [0.0], 0.0, need=True) is not None
    st = pol.stats()
    assert "width" not in st and st["width_error"].startswith("config")
    pol.configure({"width": {"widths": [4, 8]}})
    assert pol._wd is not None and pol._wd.widths == [4, 8]
    assert "width_error" not in pol.stats()
    pol.configure({"mode": "legacy"})
    assert pol._wd is None and "width" not in pol.stats()


def test_width_exception_is_isolated(monkeypatch):
    v37 = load_37()
    old = v37.VcapPolicy(P7, {"mode": "adapt"}, seed=0)
    new = vp.VcapPolicy(P7, {"mode": "adapt", "width": {"enabled": True}}, seed=0)

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(new._wd, "decide", boom)
    o1 = old.prepare_step(["a", "b"], [0.0, 0.7], 1.0)
    o2 = new.prepare_step(["a", "b"], [0.0, 0.7], 1.0)
    for x, y in zip(o1, o2):
        assert np.array_equal(x, y)
    assert new._wd is None
    st = new.stats()
    assert st["width_error"].startswith("prepare") and "boom" in st["width_error"]


# ---------------------------------------------------------------- 8. 与真 verify_cap.py 联调
def test_verify_cap_width_mode_is_legacy(tmp_path):
    torch = pytest.importorskip("torch")
    import test_vcap as tv  # 复用 r653 的假 SGLang / 假 verify 模块

    new, old, ex, holder = tv.setup_wrapped(tmp_path, 7, {"mode": "width", "tag": "W1"})
    rng = np.random.default_rng(9)
    rids = ["a", "b", "c", "d"]
    for i in range(60):
        conf, _ = tv.rand_conf(rng, 4, 7)
        old.set_live_from_confidence(conf, 4)
        live = tv.step(new, ex, conf, rids, [0.0, 0.7, 0.0, 1.0], torch.tensor([1, 2, 0, 3]), holder)
        assert torch.equal(live, old._state["live"][:4]), i
        time.sleep(0.002)
    ack = json.loads((tmp_path / "ack.json").read_text())
    assert ack["mode"] == "legacy" and ack["tag"] == "W1"
    pol = new._VC["policy"]
    assert pol._wd is not None and pol._wd.W0 == 8
    new._vc_flush(time.time())
    recs = [json.loads(x) for x in (tmp_path / "stats.jsonl").read_text().splitlines() if x.strip()]
    ws = [r["width"] for r in recs if "width" in r]
    assert ws and all(r.get("mode") == "legacy" for r in recs)
    assert sum(w["timed"] for w in ws) >= 50
    assert sum(w["by_bs"]["4"]["n"] for w in ws if "4" in w["by_bs"]) >= 50
    os.remove(tmp_path / "ctl.json")
    new._VC["next_check"] = 0.0
    conf, _ = tv.rand_conf(rng, 4, 7)
    old.set_live_from_confidence(conf, 4)
    live = tv.step(new, ex, conf, rids, [0.0] * 4, torch.tensor([0, 0, 0, 0]), holder)
    assert new._VC["active"] is False and torch.equal(live, old._state["live"][:4])


# ---------------------------------------------------------------- 9. 性能
def test_width_perf_bs16():
    P, bs = 7, 16
    rng = np.random.default_rng(23)
    rids = ["q%d" % i for i in range(bs)]
    temps = [0.0, 0.7] * 8
    data = []
    for _ in range(300):
        cum = np.cumprod(rng.uniform(0.5, 1.0, size=(bs, P)), axis=1)
        k = rng.integers(1, P + 1, size=bs)
        data.append((cum, k, np.minimum(rng.integers(0, P + 1, size=bs), k), rng.random(bs) < 0.3))

    def run(cfg):
        best = None
        for _ in range(3):
            pol = vp.VcapPolicy(P, cfg)
            t = 0.0
            t0 = time.perf_counter()
            for cum, k, acc, rec in data:
                pol.prepare_step(rids, temps, t, need=False)
                if t > 0:
                    pol.observe(rids, k, cum, rec, acc, t - 0.05)
                t += 0.05
            us = (time.perf_counter() - t0) / len(data) * 1e6
            best = us if best is None else min(best, us)
        return best

    off = run({"mode": "legacy"})
    on = run({"mode": "width", "width": W3})  # 三个宽度：最坏情况
    print("PERF width bs=16 P=7: off %.1f us/step, on %.1f us/step, +%.1f" % (off, on, on - off))
    assert on - off < 400, (off, on)

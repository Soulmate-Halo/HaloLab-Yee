"""vcap 单测（r653，随 yee/patches/mia-dsv41-vcap 发布）：legacy 逐位等价、adapt 参考实现、
策略收敛、控制文件、异常安全、性能。torch 用 pytest.importorskip。"""
import importlib.util
import json
import math
import os
import sys
import time
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PD = os.path.join(PKG, "yee", "patches", "mia-dsv41-vcap", "files")
ORIG = os.path.join(HERE, "fixtures", "mia_verify_cap_orig.py")  # Mia image /opt/dsv41/adapter/verify_cap.py
sys.path.insert(0, PD)
import vcap_policy as vp  # noqa: E402


class FakeGroup(object):
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.n_bcast = 0

    def broadcast(self, t, src=0):
        self.n_bcast += 1


def fake_sglang(group):
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["sglang.srt.distributed"].get_tp_group = lambda: group


def load(path, name, env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


def envs(tmp_path, block, spec="conf:0.1"):
    return {"DSV41_VERIFY_CAP": spec, "DSPARK_BLOCK_SIZE": block,
            "DSV41_VCAP_CTL": str(tmp_path / "ctl.json"), "DSV41_VCAP_ACK": str(tmp_path / "ack.json"),
            "DSV41_VCAP_STATS": str(tmp_path / "stats.jsonl"), "DSV41_VCAP_REQLOG": str(tmp_path / "req.jsonl")}


class Req(object):
    def __init__(self, rid, temp=0.0):
        self.rid = rid
        self.sampling_params = types.SimpleNamespace(temperature=temp)


class Outs(object):
    def __init__(self, correct_len):
        self.correct_len = correct_len


def make_verify_module(holder):
    m = types.SimpleNamespace()

    class Ep(object):
        def _accept(self, **kw):
            return kw

    class Ex(object):
        def run_non_compact(self, *, batch, draft_input, verify_ids_2d, **kw):
            return "ran"

        def accept_and_finalize(self, *a, **kw):
            return Outs(holder["acc"])

    m.DsparkVerifyEpilogue = Ep
    m.TargetVerifyExecutor = Ex
    m.accept_draft_tokens = lambda **kw: kw
    return m


def rand_conf(rng, bs, P, rec_frac=0.3):
    c = torch.from_numpy(rng.uniform(0.0, 1.0, size=(bs, P)).astype(np.float32)) ** 0.5
    rec = rng.random(bs) < rec_frac
    c[torch.from_numpy(rec)] = 1.0
    return c, rec


def step(mod, ex, conf, rids, temps, acc, holder):
    """模拟一步：propose 留下 confidence → run_non_compact → accept_and_finalize。返回本步 live。"""
    bs = conf.shape[0]
    mod._state["conf"] = conf
    batch = types.SimpleNamespace(reqs=[Req(r, t) for r, t in zip(rids, temps)])
    ex.run_non_compact(batch=batch, draft_input=None, verify_ids_2d=torch.zeros((bs, mod.STRIDE)))
    live = mod._state["live"][:bs].clone()
    holder["acc"] = acc
    ex.accept_and_finalize()
    return live


# ---------------------------------------------------------------- 1. 无控制文件：与原文件逐位一致
@pytest.mark.parametrize("block", [7, 12])
@pytest.mark.parametrize("spec", ["conf:0.1", "conf:0.25,0.25,0.03,0.06"])
def test_no_ctl_identical_to_original(tmp_path, block, spec):
    g_old, g_new = FakeGroup(), FakeGroup()
    e = envs(tmp_path, block, spec)
    fake_sglang(g_old)
    old = load(ORIG, "vc_old_%d" % block, e)
    fake_sglang(g_new)
    new = load(os.path.join(PD, "verify_cap.py"), "vc_new_%d" % block, e)
    rng = np.random.default_rng(block)
    P = block
    for i in range(1000):
        bs = int(rng.integers(1, 17))
        conf, _ = rand_conf(rng, bs, P)
        old._state["tp_group"] = g_old
        new._state["tp_group"] = g_new
        old.set_live_from_confidence(conf, bs)
        new._vc_before_step(types.SimpleNamespace(reqs=[Req(str(j)) for j in range(bs)]), bs, conf.device)
        new.set_live_from_confidence(conf, bs)
        assert torch.equal(old._state["live"][:bs], new._state["live"][:bs]), i
    assert g_old.n_bcast == g_new.n_bcast == 0  # world_size 1
    assert new._VC["active"] is False and new._VC["pending"] is None
    assert not os.path.exists(tmp_path / "ack.json")


def test_no_ctl_broadcast_count_tp4(tmp_path):
    g_old, g_new = FakeGroup(world_size=4), FakeGroup(world_size=4)
    e = envs(tmp_path, 12)
    fake_sglang(g_old)
    old = load(ORIG, "vc_old_b", e)
    fake_sglang(g_new)
    new = load(os.path.join(PD, "verify_cap.py"), "vc_new_b", e)
    rng = np.random.default_rng(3)
    for _ in range(50):
        conf, _ = rand_conf(rng, 8, 12)
        old._state["tp_group"] = g_old
        new._state["tp_group"] = g_new
        old.set_live_from_confidence(conf, 8)
        new.set_live_from_confidence(conf, 8)
    assert g_old.n_bcast == g_new.n_bcast == 50


# ---------------------------------------------------------------- 2/3. 控制文件 legacy / adapt
def setup_wrapped(tmp_path, block, ctl):
    g = FakeGroup()
    fake_sglang(g)
    e = envs(tmp_path, block)
    new = load(os.path.join(PD, "verify_cap.py"), "vc_w_%d_%s" % (block, ctl.get("mode") if ctl else "none"), e)
    old = load(ORIG, "vc_wo_%d" % block, e)
    old._state["tp_group"] = g
    holder = {"acc": None}
    vm = make_verify_module(holder)
    new.install_verify(vm)
    if ctl is not None:
        (tmp_path / "ctl.json").write_text(json.dumps(ctl), encoding="utf-8")
    return new, old, vm.TargetVerifyExecutor(), holder


def test_ctl_legacy_same_k_and_observes(tmp_path):
    new, old, ex, holder = setup_wrapped(tmp_path, 12, {"mode": "legacy", "tag": "L1"})
    rng = np.random.default_rng(5)
    rids = ["a", "b", "c", "d"]
    for i in range(40):
        conf, _ = rand_conf(rng, 4, 12)
        old.set_live_from_confidence(conf, 4)
        live = step(new, ex, conf, rids, [0.0, 0.7, 0.0, 1.0], torch.tensor([1, 2, 0, 3]), holder)
        assert torch.equal(live, old._state["live"][:4])
    ack = json.loads((tmp_path / "ack.json").read_text())
    assert ack["mode"] == "legacy" and ack["tag"] == "L1"
    pol = new._VC["policy"]
    assert pol is not None and pol.tot["new_reqs"] == 4
    assert pol.win["steps"] >= 38          # observe 已被调用（最后一步还在队列里）


def ref_k(conf, thr, krec, kadd, kmin, kmax):
    cum = torch.cumprod(conf.float().clamp(0, 1), dim=1).numpy()
    conf = conf.float().clamp(0, 1).numpy()
    out = []
    for i in range(conf.shape[0]):
        if (conf[i] >= 1.0).all():
            k = int(krec[i])
        else:
            k = 0
            for j in range(conf.shape[1]):
                if cum[i, j] >= thr[i, j]:
                    k += 1
                else:
                    break
        out.append(min(max(k + int(kadd[i]), kmin), kmax))
    return np.array(out)


@pytest.mark.parametrize("block", [7, 12])
def test_ctl_adapt_matches_reference(tmp_path, block):
    new, _, ex, holder = setup_wrapped(tmp_path, block, {"mode": "adapt", "tag": "A1", "explore": 0.2})
    rng = np.random.default_rng(7)
    rids = ["r%d" % i for i in range(8)]
    for i in range(200):
        conf, _ = rand_conf(rng, 8, block)
        live = step(new, ex, conf, rids, [0.0] * 4 + [0.7] * 4,
                    torch.from_numpy(rng.integers(0, 3, size=8)), holder)
        thr, krec, kadd = new._VC["last_out"]
        kmin, kmax = new._VC["policy"].kbounds()
        k = ref_k(conf, thr, krec, kadd, kmin, kmax)
        assert np.array_equal(live.numpy() - 1, k), i
    assert (live.numpy() - 1).max() <= block


def test_ctl_adapt_kmax(tmp_path):
    new, _, ex, holder = setup_wrapped(tmp_path, 12, {"mode": "adapt", "tag": "K", "kmax": 6, "explore": 0.0})
    rng = np.random.default_rng(9)
    for i in range(30):
        conf, _ = rand_conf(rng, 4, 12, rec_frac=0.6)
        live = step(new, ex, conf, ["a", "b", "c", "d"], [0.0] * 4, torch.tensor([6, 6, 6, 6]), holder)
        assert int(live.max()) <= 7 and int(live.min()) >= 2


# ---------------------------------------------------------------- 4. 策略
def test_policy_all_logics_off_equals_legacy():
    pol = vp.VcapPolicy(12, {"logics": {k: False for k in vp.LOGICS}, "thr": 0.1, "explore": 0.5})
    thr, krec, kadd = pol.prepare_step(["a", "b"], [0.0, 1.0], 1.0)
    assert np.all(thr == np.float32(0.1)) and np.all(krec == 12) and np.all(kadd == 0)


def test_policy_load_theta_lambda():
    pol = vp.VcapPolicy(7, {"delta_ms": 0.87, "ema": 1.0, "explore": 0.0})
    pol.prepare_step(["a", "b"], [0, 0], 100.000)
    pol.prepare_step(["a", "b"], [0, 0], 100.050)        # 第一步历时 50 ms
    cum = np.full((2, 7), 0.5)
    pol.observe(["a", "b"], [2, 2], cum, [False, False], [1, 3], 100.000)   # acc 被截成 [1,2] → 产出 5 个
    assert abs(pol.theta - 5 / 50.0) < 1e-9
    assert abs(pol.lam() - 0.87 * 0.1) < 1e-9


@pytest.mark.parametrize("beta_true", [0.5, 2.0])
def test_policy_req_beta_converges(beta_true):
    P = 12
    pol = vp.VcapPolicy(P, {"explore": 0.05, "logics": {"pos": False, "load": False, "recall": False}})
    rng = np.random.default_rng(11)
    t = 0.0
    for s in range(2000):
        t += 0.05
        thr, krec, kadd = pol.prepare_step(["x"], [0.0], t)
        conf = rng.uniform(0.6, 0.98, size=P)
        cum = np.cumprod(conf)
        k = 0
        for j in range(P):
            if cum[j] >= thr[0, j]:
                k += 1
            else:
                break
        k = min(max(k + int(kadd[0]), 1), P)
        true_surv = conf ** beta_true
        acc = 0
        for j in range(k):
            if rng.random() < true_surv[j]:
                acc += 1
            else:
                break
        pol.observe(["x"], [k], cum[None, :], [False], [acc], t)
    b = pol.reqs["x"].beta
    assert abs(b - beta_true) / beta_true < 0.25, b


def test_policy_recall_q_converges():
    P = 12
    pol = vp.VcapPolicy(P, {"explore": 0.0, "logics": {"temp": False}})
    rng = np.random.default_rng(13)
    t = 0.0
    for s in range(1500):
        t += 0.05
        thr, krec, kadd = pol.prepare_step(["c"], [0.0], t)
        k = int(krec[0])
        acc = 0
        for j in range(k):
            if rng.random() < 0.9:
                acc += 1
            else:
                break
        pol.observe(["c"], [k], np.ones((1, P)), [True], [acc], t)
    q = pol._q_of(pol.reqs["c"])
    assert abs(q - 0.9) < 0.05, q
    lam = pol.lam()
    assert int(pol.prepare_step(["c"], [0.0], t + 0.05)[1][0]) == min(P, max(1, int(math.floor(math.log(lam) / math.log(q)))))


def test_policy_temp_prior_applies_to_new_requests():
    pol = vp.VcapPolicy(7, {})
    pol.beta_b[2] = 1.7
    pol.q_b[2] = 0.6
    pol.prepare_step(["n1"], [0.8], 1.0)
    r = pol.reqs["n1"]
    assert r.bucket == 2 and abs(r.beta - 1.7) < 1e-12 and abs(pol._q_of(r) - 0.6) < 1e-12
    pol2 = vp.VcapPolicy(7, {"logics": {"temp": False}})
    pol2.beta_b[2] = 1.7
    pol2.prepare_step(["n1"], [0.8], 1.0)
    assert pol2.reqs["n1"].beta == 1.0


def test_policy_explore_rate():
    pol = vp.VcapPolicy(12, {"explore": 0.1, "explore_add": 2})
    n = 0
    for s in range(2000):
        _, _, kadd = pol.prepare_step(["a", "b", "c", "d", "e"], [0.0] * 5, s * 0.05)
        n += int((kadd > 0).sum())
        assert set(np.unique(kadd)) <= {0, 2}
    rate = n / 10000.0
    assert 0.07 < rate < 0.13, rate


def test_policy_gc_finalize_records():
    pol = vp.VcapPolicy(7, {"tag": "T"})
    pol.prepare_step(["gone"], [0.7], 0.0)
    for s in range(1, 100):
        pol.prepare_step(["stay"], [0.0], s * 0.05)
    fin = pol.drain_finished()
    assert [f["rid"] for f in fin] == ["gone"] and fin[0]["tag"] == "T" and fin[0]["bucket"] == 2
    assert "stay" in pol.reqs


# ---------------------------------------------------------------- 5. 控制文件热重读 / ack / 坏 JSON / 统计
def test_ctl_reload_ack_badjson_stats(tmp_path):
    new, _, ex, holder = setup_wrapped(tmp_path, 7, {"mode": "legacy", "tag": "t1", "stats_s": 0})
    rng = np.random.default_rng(17)

    def run(n):
        for _ in range(n):
            conf, _ = rand_conf(rng, 2, 7)
            step(new, ex, conf, ["a", "b"], [0.0, 0.7], torch.tensor([1, 1]), holder)

    run(3)
    assert json.loads((tmp_path / "ack.json").read_text())["tag"] == "t1"
    time.sleep(0.02)
    (tmp_path / "ctl.json").write_text(json.dumps({"mode": "adapt", "tag": "t2", "stats_s": 0}), encoding="utf-8")
    new._VC["next_check"] = 0.0
    run(3)
    ack = json.loads((tmp_path / "ack.json").read_text())
    assert ack["tag"] == "t2" and ack["mode"] == "adapt" and new._VC["mode"] == "adapt"
    time.sleep(0.02)
    (tmp_path / "ctl.json").write_text("{not json", encoding="utf-8")
    new._VC["next_check"] = 0.0
    run(2)
    assert new._VC["mode"] == "adapt" and new._VC["disabled"] is False
    lines = [json.loads(x) for x in (tmp_path / "stats.jsonl").read_text().splitlines()]
    assert {x["tag"] for x in lines} >= {"t1", "t2"}
    for x in lines:
        for key in ("steps", "mean_live", "mean_acc", "hist_k", "theta", "lam", "step_ms", "thr_mean",
                    "frac_rec", "beta_by_bucket", "q_by_bucket", "s_pos", "explore_n", "dropped_obs",
                    "len_mismatch", "policy_us_mean", "block", "mode", "t"):
            assert key in x
    os.remove(tmp_path / "ctl.json")
    new._VC["next_check"] = 0.0
    run(1)
    assert new._VC["active"] is False


def test_len_mismatch_falls_back(tmp_path):
    new, old, ex, holder = setup_wrapped(tmp_path, 7, {"mode": "adapt", "tag": "m"})
    conf = torch.full((3, 7), 0.9)
    new._state["conf"] = conf
    ex.run_non_compact(batch=types.SimpleNamespace(reqs=[Req("a")]), draft_input=None,
                       verify_ids_2d=torch.zeros((3, 8)))
    old.set_live_from_confidence(conf, 3)
    assert torch.equal(new._state["live"][:3], old._state["live"][:3])
    assert new._VC["policy"].win["len_mismatch"] == 1


# ---------------------------------------------------------------- 6. 异常安全
def test_policy_exception_falls_back(tmp_path):
    new, old, ex, holder = setup_wrapped(tmp_path, 12, {"mode": "adapt", "tag": "E"})
    rng = np.random.default_rng(19)
    conf, _ = rand_conf(rng, 4, 12)
    step(new, ex, conf, ["a", "b", "c", "d"], [0.0] * 4, torch.tensor([0, 0, 0, 0]), holder)

    def boom(*a, **kw):
        raise RuntimeError("boom")

    new._VC["policy"].prepare_step = boom
    for _ in range(5):
        conf, _ = rand_conf(rng, 4, 12)
        old.set_live_from_confidence(conf, 4)
        live = step(new, ex, conf, ["a", "b", "c", "d"], [0.0] * 4, torch.tensor([0, 0, 0, 0]), holder)
        assert torch.equal(live, old._state["live"][:4])
    assert new._VC["disabled"] is True
    assert json.loads((tmp_path / "ack.json").read_text())["mode"] == "failed"


# ---------------------------------------------------------------- 7. 性能
def test_policy_perf_bs16_p12():
    P, bs = 12, 16
    pol = vp.VcapPolicy(P, {})
    rng = np.random.default_rng(23)
    rids = ["q%d" % i for i in range(bs)]
    temps = [0.0, 0.7] * 8
    t = 0.0
    n = 400
    data = []
    for s in range(n):
        cum = np.cumprod(rng.uniform(0.5, 1.0, size=(bs, P)), axis=1)
        k = rng.integers(1, P + 1, size=bs)
        data.append((cum, k, np.minimum(rng.integers(0, P, size=bs), k), rng.random(bs) < 0.3))
    t0 = time.perf_counter()
    for cum, k, acc, rec in data:
        t += 0.05
        pol.prepare_step(rids, temps, t)
        pol.observe(rids, k, cum, rec, acc, t)
    us = (time.perf_counter() - t0) / n * 1e6
    print("PERF prepare+observe bs=16 P=12: %.1f us/step" % us)
    assert us < 1000, us  # sanity bound; r653 servers: 94~300 us


# ---------------------------------------------------------------- CUDA 路径（pinned + event）
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_adapt_path(tmp_path):
    new, _, ex, holder = setup_wrapped(tmp_path, 12, {"mode": "adapt", "tag": "C", "explore": 0.1})
    rng = np.random.default_rng(29)
    rids = ["g%d" % i for i in range(8)]
    for i in range(60):
        conf, _ = rand_conf(rng, 8, 12)
        conf = conf.cuda()
        new._state["conf"] = conf
        batch = types.SimpleNamespace(reqs=[Req(r, 0.7) for r in rids])
        ex.run_non_compact(batch=batch, draft_input=None, verify_ids_2d=torch.zeros((8, 13), device="cuda"))
        live = new._state["live"][:8].clone()
        holder["acc"] = torch.from_numpy(rng.integers(0, 3, size=8)).to(torch.int32).cuda()
        ex.accept_and_finalize()
        torch.cuda.synchronize()
        thr, krec, kadd = new._VC["last_out"]
        kmin, kmax = new._VC["policy"].kbounds()
        assert np.array_equal(live.cpu().numpy() - 1, ref_k(conf.cpu(), thr, krec, kadd, kmin, kmax)), i
    assert new._VC["policy"].win["steps"] + new._VC["policy"].tot.get("dropped_obs", 0) >= 50
    assert new._VC["h2d"][0]["thr"].is_pinned()

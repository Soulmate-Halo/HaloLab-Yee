"""CPU tests of yee/integrations/expert_budget.py (reference semantics, control parsing, import hygiene)."""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "yee", "integrations", "expert_budget.py")


def _load():
    spec = importlib.util.spec_from_file_location("expert_budget_t", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eb = _load()
E, K = 384, 6


def _case(seed, M, kind="uniform"):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(M, E, generator=g)
    if kind == "hot":
        scores[:, torch.randperm(E, generator=g)[:20]] += 1.0
    if kind == "corr":
        scores = 0.7 * torch.rand(1, E, generator=g) + 0.3 * scores
    ids = scores.topk(K, dim=1).indices.to(torch.int32)
    w = torch.rand(M, K, generator=g) + 0.05
    w = w / w.sum(dim=1, keepdim=True) * 1.5
    if kind == "invalid":
        ids[torch.rand(M, K, generator=g) < 0.15] = -1
        ids[0, :] = -1
        if M > 2:
            ids[1, 2] = E + 3
    return ids, w


CASES = [(s, m, k) for s, (m, k) in enumerate([(1, "uniform"), (8, "corr"), (8, "invalid"), (40, "hot"),
                                                (64, "uniform"), (128, "corr"), (300, "hot"), (17, "invalid")])]


@pytest.mark.parametrize("seed,M,kind", CASES)
def test_identity_when_disarmed(seed, M, kind):
    ids, w = _case(seed, M, kind)
    ei, ew = eb.ep_remap_reference(ids, w, E)
    for p, kcap, armed in [(0.5, 3, 0), (1.0, 4096, 1), (1.0, E, 1)]:
        oi, ow, info = eb.budget_reference(ids, w, E, p, kcap, armed)
        assert torch.equal(oi, ei)
        assert torch.equal(ow, ew)
        assert info["mass_kept"] == info["mass_total"] and info["forced_rows"] == 0


def _mass(ids, w):
    ok = (ids >= 0) & (ids < E)
    wq = torch.floor(w.clamp(0, eb.WMAX) * eb.QSCALE + 0.5).to(torch.int64).clamp_min(1)
    wq = torch.where(ok, wq, torch.zeros_like(wq))
    m = torch.zeros(E, dtype=torch.int64)
    m.index_add_(0, torch.where(ok, ids.long(), torch.zeros_like(ids.long())).reshape(-1), wq.reshape(-1))
    return m


@pytest.mark.parametrize("seed,M,kind", CASES)
@pytest.mark.parametrize("p", [0.97, 0.9, 0.8, 0.6])
def test_coverage_minimal(seed, M, kind, p):
    ids, w = _case(seed, M, kind)
    oi, ow, info = eb.budget_reference(ids, w, E, p, 4096, 1)
    mass = _mass(ids, w)
    total = int(mass.sum())
    t = info["t"]
    keep = (mass >= t) & (mass > 0)
    kept_mass = int(mass[keep].sum())
    assert kept_mass == info["mass_kept"]
    assert kept_mass >= p * total * (1 - 1e-6)
    # dropping the lightest kept tier must fall below the target
    tier = int(mass[keep].min())
    if (mass[keep] > tier).any():
        assert int(mass[mass > tier].sum()) < p * total * (1 + 1e-6)


@pytest.mark.parametrize("seed,M,kind", CASES)
@pytest.mark.parametrize("kcap", [64, 24, 12, 4, 0])
def test_kcap(seed, M, kind, kcap):
    ids, w = _case(seed, M, kind)
    oi, ow, info = eb.budget_reference(ids, w, E, 1.0, kcap, 1)
    mass = _mass(ids, w)
    keep = (mass >= info["t"]) & (mass > 0)
    assert int(keep.sum()) <= kcap


@pytest.mark.parametrize("seed,M,kind", CASES)
@pytest.mark.parametrize("p,kcap", [(0.9, 4096), (0.6, 4096), (1.0, 4), (0.8, 12), (1.0, 0)])
def test_rows_and_ids(seed, M, kind, p, kcap):
    ids, w = _case(seed, M, kind)
    oi, ow, info = eb.budget_reference(ids, w, E, p, kcap, 1)
    ok = (ids >= 0) & (ids < E)
    rv = ok.any(dim=1)
    nz = (ow > 0).any(dim=1)
    assert bool((nz | ~rv).all()), "a row with valid entries kept nothing"
    rs_in = torch.where(ok, w, torch.zeros_like(w)).sum(dim=1)
    rs_out = ow.sum(dim=1)
    assert torch.allclose(rs_out[rv], rs_in[rv], rtol=1e-6, atol=0)
    assert bool(((oi >= 0) & (oi < E)).all())
    mass = _mass(ids, w)
    keep = (mass >= info["t"]) & (mass > 0)
    used = set(torch.unique(oi).tolist())
    forced = set(oi[ow > 0].tolist()) - set(torch.nonzero(keep).flatten().tolist())
    allowed = set(torch.nonzero(keep).flatten().tolist()) | forced
    if not keep.any():
        allowed |= {int(torch.nonzero(mass > 0).flatten()[0])} if (mass > 0).any() else {0}
    assert used <= allowed
    assert info["uniq_after"] <= max(info["uniq_before"], 1)
    assert info["uniq_after"] == len(used)


def test_dropped_slots_point_at_kept_experts_with_zero_weight():
    ids, w = _case(3, 64, "hot")
    oi, ow, info = eb.budget_reference(ids, w, E, 0.7, 4096, 1)
    assert info["mass_kept"] < info["mass_total"]
    changed = oi != ids
    assert bool(changed.any())
    assert bool((ow[changed] == 0).all())
    for r in range(oi.shape[0]):
        live = set(oi[r][ow[r] > 0].tolist())
        assert set(oi[r].tolist()) <= live


def test_parse_ctrl():
    assert eb.parse_ctrl({"seq": 1, "armed": 1, "p": 0.9, "kcap": 24}) == (1, 1, 0.9, 24, None)
    assert eb.parse_ctrl({"seq": 2}) == (2, 0, 1.0, 4096, None)
    assert eb.parse_ctrl({"seq": 3, "armed": True, "p": 1}) == (3, 1, 1.0, 4096, None)
    assert eb.parse_ctrl({"seq": 4, "armed": 1, "p": 0.9, "at_calls": 123456}) == (4, 1, 0.9, 4096, 123456)
    for bad in [None, [], {}, {"seq": "1"}, {"seq": True}, {"seq": 1, "armed": 2}, {"seq": 1, "p": 0},
                {"seq": 1, "p": 1.5}, {"seq": 1, "p": "0.9"}, {"seq": 1, "kcap": -1}, {"seq": 1, "kcap": 2.5},
                {"seq": 1, "kcap": True}, {"seq": 1, "at_calls": -1}, {"seq": 1, "at_calls": 1.5},
                {"seq": 1, "at_calls": True}, {"seq": 1, "at_calls": 1 << 62}]:
        assert eb.parse_ctrl(bad) is None, bad


def test_bucket():
    assert [eb.bucket_of(m) for m in (1, 8, 9, 64, 128, 129, 512)] == [1, 1, 2, 8, 16, 16, 16]


def test_import_does_not_pull_triton_or_torch():
    code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('x',%r);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);print('triton' in sys.modules, 'torch' in sys.modules)" % PATH)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="needs CUDA + triton")
def test_gpu_selftest():
    assert eb.selftest(n=60) == 0


# --- 4.1.1 regression: the serving host sets torch's default dtype to bfloat16 while the model is built,
# so every implicit torch.tensor() in the switch check became bf16 and 0.8 turned into 0.80078125.
def test_new_ctrl_is_float32_under_bf16_default_dtype():
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        ctrl, sched = eb.new_ctrl(0.8, 4096, 1, "cpu")
        assert ctrl.dtype == torch.float32
        assert float(ctrl[0]) == float(torch.tensor(0.8, dtype=torch.float32))
        assert sched.dtype == torch.int64
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="needs CUDA + triton")
@pytest.mark.parametrize("K", [6, 8])
def test_quick_check_passes_under_bf16_default_dtype(K):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        assert eb._quick_check(torch.device("cuda:0"), E, K) is None
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="needs CUDA + triton")
def test_arm_with_initial_ctrl_already_armed():
    old_dtype = torch.get_default_dtype()
    old_ctrl = eb._STATE["ctrl"]
    old_reg = list(eb._REG)
    old_start = eb._start_thread

    class _Geom:
        E = E
        topk = 6
        device = torch.device("cuda:0")

    try:
        torch.set_default_dtype(torch.bfloat16)
        eb._start_thread = lambda *a, **kw: None
        eb._STATE["ctrl"] = (0.9, 4096, 1)
        eb._arm(_Geom())  # must not raise: the warm-up identity check only holds for an unarmed ctrl
    finally:
        torch.set_default_dtype(old_dtype)
        eb._start_thread = old_start
        eb._STATE["ctrl"] = old_ctrl
        eb._REG[:] = old_reg

"""KAD threshold mode: the width of the whole batch is decided at once by a threshold.

The threshold strategy keeps a conditional acceptance rate ``g_j`` per request (the probability
that the position ``j`` is accepted when the positions ``1..j-1`` were accepted).  The survival
rates ``S_j = g_1 * ... * g_j`` give the threshold ``tau = c * R``: the position ``j`` of a
request is sent for verification when ``S_ij > tau``.  The widths of the requests of the batch
are solved together (the throughput of the batch depends on all of them).

The reference implementation is ``.diagnostics/r648/s3/replay_s3.py`` (``solve_threshold``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vllm_d_recall.integrations import vllm_kad as vk

KMAX = 12
BASE = 40.0
C = 0.49
P = 0.30

# The survival rates the default priors are derived from.
SURVIVAL = {
    "T0": (0.605, 0.332, 0.177, 0.101, 0.069),
    "T(0,0.5]": (0.577, 0.302, 0.144, 0.074, 0.046),
    "T(0.5,0.9]": (0.577, 0.302, 0.144, 0.074, 0.046),
    "T>0.9": (0.577 * 0.9, 0.302 * 0.9, 0.144 * 0.9, 0.074 * 0.9, 0.046 * 0.9),
}


# --------------------------------------------------------------------------- the reference solver
def _replay_s3():
    """The reference solver of the offline replay (skipped when the folder is not there)."""
    folder = Path(__file__).resolve().parents[2] / "s3"
    if not (folder / "replay_s3.py").exists():
        return None
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        import replay_s3
    except Exception:  # noqa: BLE001  numpy or the shadow module is missing
        return None
    return replay_s3


# --------------------------------------------------------------------------- fakes
class FakeSP:
    """The sampling parameters of a request."""

    def __init__(self, temperature):
        self.temperature = temperature
        self.top_p = 1.0
        self.top_k = 0


class FakeReq:
    """A scheduled request of the fake scheduler."""

    def __init__(self, rid, spec, temp=0.0, structured=False):
        self.request_id = rid
        self.spec_token_ids = list(spec)
        self.num_computed_tokens = 10
        self.num_prompt_tokens = 8
        self.sampling_params = FakeSP(temp) if temp is not None else None
        self.use_structured_output = structured
        self.next_decode_eligible_step = 0
        self.is_prefill_chunk = False


class FakeSched:
    """The fake scheduler: the requests, the draft budget and the step counter."""

    def __init__(self, requests, kmax=KMAX):
        self.requests = {r.request_id: r for r in requests}
        self.running = list(requests)
        self.num_spec_tokens = kmax
        self.current_step = 0


class FakeSO:
    """The fake SchedulerOutput."""

    def __init__(self, num_scheduled_tokens, spec_tokens=None):
        self.num_scheduled_tokens = dict(num_scheduled_tokens)
        self.scheduled_spec_decode_tokens = dict(spec_tokens or {})


class FakeMRO:
    """The fake ModelRunnerOutput: the request index and the sampled tokens."""

    def __init__(self, idx_map, sampled):
        self.req_id_to_index = dict(idx_map)
        self.sampled_token_ids = [list(s) for s in sampled]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def kad_env(monkeypatch, tmp_path):
    """KAD on, the shadow off, and a clean state."""
    monkeypatch.setenv("VLLM_D_RECALL", "1")
    monkeypatch.setenv("VLLM_D_RECALL_KAD", "1")
    monkeypatch.setenv("VLLM_D_RECALL_DIR", str(tmp_path / "kad"))
    monkeypatch.setenv("VLLM_D_RECALL_KAD_LOG_S", "0")
    monkeypatch.setenv("VLLM_D_RECALL_KAD_RELOAD_S", "0")
    monkeypatch.delenv("VLLM_D_RECALL_SHADOW", raising=False)
    vk.reset()
    yield tmp_path
    vk.reset()


def _write_policy(tmp_path, obj):
    """Write the policy file and return its path."""
    path = tmp_path / "kad" / "kad_policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _run_batch(reqs, kmax=KMAX, step=0):
    """Drive one scheduling step; returns the widths of the requests."""
    sched = FakeSched(reqs, kmax=kmax)
    sched.current_step = step
    so = FakeSO({r.request_id: len(r.spec_token_ids) + 1 for r in reqs})
    vk.post_schedule(sched, so)
    return dict(getattr(so, vk.ATTR_SO)["w"])


def _as_arrays(S_list):
    """The survival rates as the numpy arrays the reference solver wants."""
    import numpy as np

    return [np.asarray(s, dtype=float) for s in S_list]


# --------------------------------------------------------------------------- the tests
def test_default_priors_round_trip():
    """The default priors are the conditional acceptance rates of the default survival rates."""
    priors = vk.thr_default_priors()
    assert sorted(priors) == sorted(SURVIVAL)
    for bucket, surv in SURVIVAL.items():
        g = priors[bucket]
        assert len(g) == KMAX
        s = vk.thr_survival(g)
        for j, want in enumerate(surv):
            assert abs(s[j] - want) < 1e-9, (bucket, j, s[j], want)
        # the positions after the known ones are extrapolated geometrically
        assert all(0.0 < x < 1.0 for x in s[len(surv):])
        assert all(s[j] <= s[j - 1] for j in range(1, KMAX))


def test_policy_defaults_and_tolerance(kad_env, capsys):
    """The threshold fields of a policy: the defaults, the invalid values and the priors."""
    pol = vk.parse_policy({"mode": "threshold"})
    assert pol["mode"] == "threshold"
    assert pol["c_ms"] == 0.49 and pol["base_ms"] == 40.0 and pol["p_ms"] == 0.30
    assert pol["wmin"] == 1 and pol["wmax"] == 12 and pol["ema"] == 0.2 and pol["budget"] == 0
    assert sorted(pol["priors"]) == sorted(SURVIVAL)
    # an invalid value falls back to the default and is logged
    capsys.readouterr()
    pol = vk.parse_policy({"mode": "threshold", "c_ms": "x", "ema": 5.0, "wmin": -3, "budget": -1})
    assert pol["c_ms"] == 0.49 and pol["ema"] == 0.2 and pol["wmin"] == 1 and pol["budget"] == 0
    out = capsys.readouterr().out
    assert "c_ms" in out and "ema" in out
    # a broken prior falls back to the default one
    pol = vk.parse_policy({"mode": "threshold", "priors": {"T0": [0.5, 0.5]}})
    assert abs(pol["priors"]["T0"][0] - vk.thr_default_priors()["T0"][0]) < 1e-12


def test_the_temperature_buckets():
    """The temperature of a request gives the bucket of its prior."""
    assert vk.thr_bucket(0.0) == "T0"
    assert vk.thr_bucket(0.3) == "T(0,0.5]"
    assert vk.thr_bucket(0.7) == "T(0.5,0.9]"
    assert vk.thr_bucket(1.0) == "T>0.9"
    assert vk.thr_bucket(None) == "T0"
    assert vk.thr_bucket("x") == "T0"


def test_homogeneous_batch_matches_the_offline_solver(kad_env):
    """A homogeneous batch: the widths are the ones of the offline replay (N = 1, 4, 16)."""
    replay = _replay_s3()
    if replay is None:
        pytest.skip("the offline replay (s3/replay_s3.py) is not available")
    _write_policy(kad_env, {"mode": "threshold"})
    S = vk.thr_survival(vk.thr_default_priors()["T0"])
    for n in (1, 4, 16):
        vk.reset()
        reqs = [FakeReq("r%d" % i, range(KMAX), temp=0.0) for i in range(n)]
        widths = _run_batch(reqs)
        want = replay.solve_threshold(_as_arrays([S] * n), BASE, C, P)
        assert [widths["r%d" % i] for i in range(n)] == list(want), n


def test_mixed_batch(kad_env):
    """A mixed batch: the copy requests are much wider and the new ones do not grow with N."""
    _write_policy(kad_env, {"mode": "threshold", "priors": {"T(0.5,0.9]": [0.95] * KMAX}})
    new = FakeReq("new", range(KMAX), temp=0.0)  # the T0 prior: the new content
    copy = FakeReq("copy", range(KMAX), temp=0.7)  # the T(0.5,0.9] prior: 0.95 everywhere
    vk.reset()
    widths = _run_batch([copy, new])
    assert widths["copy"] > widths["new"] + 3, widths
    small = widths["new"]
    vk.reset()
    reqs = [copy] + [FakeReq("new%d" % i, range(KMAX), temp=0.0) for i in range(15)]
    widths = _run_batch(reqs)
    assert widths["copy"] > widths["new0"] + 3, widths
    # the width of the new requests does not grow when the batch grows
    assert widths["new0"] <= small, (small, widths["new0"])


def test_ingest_updates_the_rates_position_by_position(kad_env):
    """The ingest: the accepted positions move to 1, the rejected one to 0, the others stay."""
    _write_policy(kad_env, {"mode": "threshold", "priors": {"T0": [0.5] * KMAX}})
    req = FakeReq("r", range(KMAX), temp=0.0)
    sched = FakeSched([req])
    so = FakeSO({"r": KMAX + 1}, {"r": [1, 2, 3, 4, 5]})  # v = 5
    vk.post_schedule(sched, so)
    mro = FakeMRO({"r": 0}, [[1, 2, 3]])  # three sampled tokens: a = 2
    vk.ingest(sched, so, mro)
    g = vk._state["thr"]["r"]["g"]
    assert abs(g[0] - 0.6) < 1e-12  # position 1: accepted
    assert abs(g[1] - 0.6) < 1e-12  # position 2: accepted
    assert abs(g[2] - 0.4) < 1e-12  # position 3: rejected
    assert abs(g[3] - 0.5) < 1e-12  # position 4: not observed
    assert abs(g[4] - 0.5) < 1e-12  # position 5: not observed
    assert all(abs(x - 0.5) < 1e-12 for x in g[5:])
    # the rates stay inside the bounds
    for _ in range(50):
        vk.ingest(sched, so, FakeMRO({"r": 0}, [[1, 2, 3]]))
    g = vk._state["thr"]["r"]["g"]
    assert all(0.001 <= x <= 0.999 for x in g)


def test_the_ladder_does_not_update_the_rates(kad_env):
    """The ladder mode: the rates of the threshold mode are not touched."""
    _write_policy(kad_env, {"mode": "ladder"})
    req = FakeReq("r", range(KMAX), temp=0.0)
    sched = FakeSched([req])
    so = FakeSO({"r": KMAX + 1}, {"r": [1, 2, 3, 4, 5]})
    vk.post_schedule(sched, so)
    vk.ingest(sched, so, FakeMRO({"r": 0}, [[1, 2, 3]]))
    assert "r" not in vk._state["thr"]


def test_the_budget_caps_the_batch(kad_env):
    """The budget: the sum of (w + 1) fits the budget and no request goes below wmin."""
    _write_policy(kad_env, {"mode": "threshold", "budget": 40, "wmin": 1})
    reqs = [FakeReq("r%d" % i, range(KMAX), temp=0.0) for i in range(8)]
    widths = _run_batch(reqs)
    assert sum(widths[r.request_id] + 1 for r in reqs) <= 40, widths
    assert all(widths[r.request_id] >= 1 for r in reqs)
    # a tighter budget still holds (the floor is n * (wmin + 1) = 16)
    _write_policy(kad_env, {"mode": "threshold", "budget": 20, "wmin": 1})
    vk.reset()
    widths = _run_batch(reqs)
    assert sum(widths[r.request_id] + 1 for r in reqs) <= 20, widths
    assert all(widths[r.request_id] >= 1 for r in reqs)
    # the budget drops the positions with the smallest survival rates first
    _write_policy(kad_env, {"mode": "threshold", "budget": 20, "wmin": 1,
                             "priors": {"T0": [0.9] * KMAX}})
    vk.reset()
    widths = _run_batch(reqs)
    assert sum(widths[r.request_id] + 1 for r in reqs) <= 20, widths


def test_the_policy_switches_while_the_server_runs(kad_env):
    """The policy file: ladder, then threshold, then ladder again."""
    path = _write_policy(kad_env, {"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12})
    widths = _run_batch([FakeReq("a", range(KMAX), temp=0.0)])
    assert widths["a"] == 5, widths
    path.write_text(json.dumps({"mode": "threshold"}), encoding="utf-8")
    widths = _run_batch([FakeReq("b", range(KMAX), temp=0.0)])
    assert widths["b"] != 5, widths
    assert 1 <= widths["b"] <= KMAX
    path.write_text(json.dumps({"mode": "ladder", "base": 9, "kmin": 9, "kmax": 12, "pad": "x"}),
                    encoding="utf-8")
    widths = _run_batch([FakeReq("c", range(KMAX), temp=0.0)])
    assert widths["c"] == 9, widths
    assert vk.snapshot()["errors"] == 0


def test_a_broken_threshold_falls_back_to_the_ladder(kad_env, monkeypatch, capsys):
    """A broken threshold: the widths of the ladder are used and the error is logged once."""
    _write_policy(kad_env, {"mode": "threshold"})

    def boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(vk, "thr_solve_batch", boom)
    reqs = [FakeReq("r", range(KMAX), temp=0.0)]
    widths = _run_batch(reqs)
    assert widths["r"] == 5, widths  # the ladder default
    out = capsys.readouterr().out
    assert out.count("threshold failed") == 1
    _run_batch(reqs)
    assert capsys.readouterr().out.count("threshold failed") == 0


def test_the_ladder_widths_are_unchanged(kad_env):
    """The ladder mode: the widths are the ones of the ladder, byte for byte."""
    _write_policy(kad_env, {"mode": "ladder", "base": 5, "kmin": 5, "kmax": 12})
    reqs = [FakeReq("r%d" % i, range(KMAX), temp=0.0) for i in range(3)]
    reqs.append(FakeReq("s", range(KMAX), temp=0.0, structured=True))
    widths = _run_batch(reqs)
    assert widths == {"r0": 5, "r1": 5, "r2": 5, "s": KMAX}, widths
    # the fixed mode is untouched as well
    _write_policy(kad_env, {"mode": "fixed", "fixed": 7})
    vk.reset()
    widths = _run_batch(reqs)
    assert widths == {"r0": 7, "r1": 7, "r2": 7, "s": KMAX}, widths

"""Plain-assert tests for yee.integrations.vllm_dspark. Run: python tests/test_vllm_hook.py

A CPU simulation of vLLM's V2 runner around ``DFlashSpeculator.propose`` (same signature, same state layout:
UVA ``all_token_ids`` with a numpy view, ``total_len.gpu``, ``last_sampled_tokens [max_reqs, 1]``, ``prompt_len.np``,
``index_to_req_id``, ``sampler.sampling_states.temperature.np``). The fake target model is greedy and has a
fixed reply per prompt; verification accepts the longest matching draft prefix plus one bonus id, exactly
like greedy rejection sampling. When the real vLLM is importable (the vllm-dsv41-x86 image) the last test
installs the hook on the real DFlashSpeculator / GPUModelRunner classes.
"""
from __future__ import annotations

import inspect
import os
import shutil
import sys
import tempfile
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yee import core as rc  # noqa: E402
from yee.integrations import vllm_spec as vd  # noqa: E402

RESULTS = []
GAMMA = 5
MAXR = 8
MAXLEN = 4096
JUNK = 3  # the fake drafter always proposes this id (never right)


def test(fn):
    RESULTS.append(fn)
    return fn


def rnd(n, seed, lo=10, hi=120000):
    return np.random.default_rng(seed).integers(lo, hi, size=n, dtype=np.int64)


# ------------------------------------------------------------------------------------------- fakes
class FakeSpeculator:
    """Same propose() signature as vLLM 0.28.1rc1 DFlashSpeculator; drafts are always JUNK."""

    def __init__(self):
        self.draft_tokens = torch.zeros(MAXR, GAMMA, dtype=torch.int64)
        self.draft_token_confidence_probs = torch.zeros(MAXR, GAMMA, dtype=torch.float32)

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
        n = input_batch.num_reqs
        self.draft_tokens[:n] = JUNK
        self.draft_token_confidence_probs[:n] = 0.25
        return self.draft_tokens[:n]


class _Uva:
    def __init__(self, shape, dtype):
        self.np = np.zeros(shape, dtype=dtype)
        self.cpu = torch.from_numpy(self.np)


class _Staged:
    def __init__(self, shape, dtype, uva=False):
        if uva:
            self._uva_buf = _Uva(shape, dtype)
            self.gpu = self._uva_buf.cpu
        else:
            self.gpu = torch.zeros(shape, dtype=torch.int32)


class _Backed:
    def __init__(self, n, dtype):
        self.np = np.zeros(n, dtype=dtype)


class FakeReqStates:
    def __init__(self):
        self.all_token_ids = _Staged((MAXR, MAXLEN), np.int32, uva=True)
        self.total_len = _Staged(MAXR, np.int32)
        self.prompt_len = _Backed(MAXR, np.int32)
        self.last_sampled_tokens = torch.zeros(MAXR, 1, dtype=torch.int64)
        self.draft_tokens = torch.zeros(MAXR, GAMMA, dtype=torch.int64)
        self.req_id_to_index = {}
        self.index_to_req_id = {}
        self.free = list(range(MAXR))[::-1]

    def add(self, rid, prompt):
        idx = self.free.pop()
        self.req_id_to_index[rid] = idx
        self.index_to_req_id[idx] = rid
        self.prompt_len.np[idx] = len(prompt)
        self.all_token_ids._uva_buf.np[idx, :len(prompt)] = prompt
        self.total_len.gpu[idx] = len(prompt)
        self.draft_tokens[idx] = 0
        return idx

    def remove(self, rid):
        idx = self.req_id_to_index.pop(rid)
        self.index_to_req_id.pop(idx)
        self.free.append(idx)


class FakeRunner:
    def __init__(self, temp=0.0):
        self.req_states = FakeReqStates()
        temps = _Backed(MAXR, np.float32)
        temps.np[:] = temp
        self.sampler = types.SimpleNamespace(sampling_states=types.SimpleNamespace(temperature=temps))
        self.speculator = FakeSpeculator()
        self.finished = []

    def finish_requests(self, scheduler_output):
        for rid in sorted(set(scheduler_output.finished_req_ids) | set(scheduler_output.preempted_req_ids)):
            self.req_states.remove(rid)
        self.finished.append(scheduler_output)


class Batch:
    def __init__(self, rids, rs):
        self.req_ids = list(rids)
        self.num_reqs = len(rids)
        self.idx_mapping_np = np.array([rs.req_id_to_index[r] for r in rids], dtype=np.int32)
        self.idx_mapping = torch.from_numpy(self.idx_mapping_np.copy())
        self.has_structured_output_reqs = False


class Sim:
    """Drives requests through prefill + decode steps with the (wrapped) propose."""

    def __init__(self, runner, propose):
        self.r = runner
        self.propose = propose
        self.replies = {}
        self.out = {}
        self.max_new = {}
        self.accepted = []  # accepted draft ids per decode step and request

    def add(self, rid, prompt, reply):
        rs = self.r.req_states
        idx = rs.add(rid, prompt)
        self.replies[rid] = np.asarray(reply, dtype=np.int64)
        self.out[rid] = []
        self.max_new[rid] = len(reply)
        # prefill step: the target emits the first reply id; no drafts were verified
        self._commit(rid, idx, [int(reply[0])])

    def _commit(self, rid, idx, ids):
        rs = self.r.req_states
        T = int(rs.total_len.gpu[idx])
        ids = ids[: self.max_new[rid] - len(self.out[rid])]
        rs.all_token_ids._uva_buf.np[idx, T:T + len(ids)] = ids
        rs.total_len.gpu[idx] = T + len(ids)
        rs.last_sampled_tokens[idx, 0] = ids[-1]
        self.out[rid].extend(ids)

    def verify(self, rid):
        rs = self.r.req_states
        idx = rs.req_id_to_index[rid]
        R = self.replies[rid]
        k = len(self.out[rid])
        d = rs.draft_tokens[idx].tolist()
        acc = 0
        while acc < GAMMA and k + acc < len(R) and d[acc] == int(R[k + acc]):
            acc += 1
        ids = [int(x) for x in R[k:k + acc]]
        if k + acc < len(R):
            ids.append(int(R[k + acc]))  # bonus
        self.accepted.append(acc)
        self._commit(rid, idx, ids)

    def step(self, rids, verify=True, dummy=False):
        rs = self.r.req_states
        if verify:
            for rid in rids:
                self.verify(rid)
        b = Batch(rids, rs)
        out = self.propose(self.r.speculator, b, None, None, None, None, None, None, rs.last_sampled_tokens,
                           None, None, None, dummy_run=dummy)
        rs.draft_tokens[torch.from_numpy(b.idx_mapping_np.astype(np.int64))] = out
        return out

    def done(self, rid):
        return len(self.out[rid]) >= self.max_new[rid]

    def run(self, rids, max_steps=10000):
        steps = 0
        live = list(rids)
        self.step(live, verify=False)  # the prefill step's propose
        while live and steps < max_steps:
            self.step(live)
            steps += 1
            fin = [r for r in live if self.done(r)]
            if fin:
                self.r.finish_requests(types.SimpleNamespace(finished_req_ids=set(fin), preempted_req_ids=set()))
                live = [r for r in live if r not in fin]
        return steps


class NonGreedy:
    """RECALL_ACCEL_NONGREEDY set to ``v`` inside the block."""

    def __init__(self, v):
        self.v = v

    def __enter__(self):
        self.old = os.environ.get("RECALL_ACCEL_NONGREEDY")
        os.environ["RECALL_ACCEL_NONGREEDY"] = self.v
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("RECALL_ACCEL_NONGREEDY", None)
        else:
            os.environ["RECALL_ACCEL_NONGREEDY"] = self.old


class Env:
    """Recall on, memory backend, record in a temp dir; resets the hook state."""

    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="vllm_recall_t_")
        self.keys = {"VLLM_D_RECALL": "1", "VLLM_D_RECALL_DIR": os.path.join(self.d, "rec"),
                     "VLLM_D_RECALL_GIB": "0.01", "VLLM_D_RECALL_BACKEND": "memory",
                     "VLLM_D_RECALL_TOKENIZER_DIR": self.d}
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ.update(self.keys)
        rc._ENGINE = None
        vd.reset_state()
        vd._event_factory = vd._default_event_factory
        return self

    def __exit__(self, *a):
        eng = vd._state["engine"]
        if eng is not None:
            eng.close()
        rc._ENGINE = None
        vd.reset_state()
        vd._event_factory = vd._default_event_factory
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.d, ignore_errors=True)


def make(temp=0.0):
    runner = FakeRunner(temp)
    vd._state["runner"] = runner
    wrapped = vd._make_propose(FakeSpeculator.propose)
    # finish_requests goes through the same wrapper the real patch installs
    mod = types.ModuleType("xy_fake_runner_mod")
    mod.GPUModelRunner = type("GPUModelRunner", (), {
        "__init__": lambda self: None, "finish_requests": FakeRunner.finish_requests})
    vd._patch_runner_module(mod)
    runner.finish_requests = types.MethodType(mod.GPUModelRunner.finish_requests, runner)
    return runner, Sim(runner, wrapped)


def flush():
    eng = vd._state["engine"]
    if eng is not None:
        eng.flush()


# ------------------------------------------------------------------------------------------- tests
@test
def test_signature_kept():
    w = vd._make_propose(FakeSpeculator.propose)
    assert tuple(inspect.signature(w).parameters) == vd.PROPOSE_PARAMS, tuple(inspect.signature(w).parameters)
    assert w.__recall_wrapped__ and w.__wrapped__ is FakeSpeculator.propose


@test
def test_second_pass_copies_and_output_unchanged():
    with Env():
        runner, sim = make()
        prompt, reply = rnd(80, 1), rnd(400, 2)
        sim.add("a", prompt, reply)
        s1 = sim.run(["a"])
        flush()
        assert sim.out["a"] == reply.tolist()
        acc1 = sum(sim.accepted)
        sim.accepted.clear()
        sim.add("b", prompt, reply)
        s2 = sim.run(["b"])
        assert sim.out["b"] == reply.tolist(), "greedy output changed"
        acc2 = sum(sim.accepted)
        assert acc1 == 0, acc1  # the junk drafter never lands anything
        assert acc2 > 250, (acc2, s2)
        assert s2 < s1 / 3, (s1, s2)
        st = vd._counts
        assert st["used"] > 0 and st["plan_rows"] > 0, dict(st)
        print("    pass1 steps=%d accepted=%d | pass2 steps=%d accepted=%d used=%d fallback=%d"
              % (s1, acc1, s2, acc2, st["used"], st["fallback"]))


@test
def test_alignment_with_three_step_cpu_lag():
    """The CPU view trails by 3 steps (snapshot of step s is read at step s+3) (events complete late): copies must still line up."""
    class LagEvent:
        live = []

        def __init__(self):
            self.age = 0
            LagEvent.live.append(self)

        def record(self):
            for e in LagEvent.live:
                e.age += 1

        def query(self):
            return self.age > 2

        def synchronize(self):
            self.age = 99

    with Env():
        runner, sim = make()
        prompt, reply = rnd(64, 3), rnd(500, 4)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        vd._event_factory = lambda dev: LagEvent()
        sim.accepted.clear()
        sim.add("b", prompt, reply)
        s2 = sim.run(["b"])
        assert sim.out["b"] == reply.tolist()
        acc = sum(sim.accepted)
        assert acc > 300, (acc, s2, dict(vd._counts))
        # every merged row was accounted as used (aligned), none as misaligned fallback
        assert vd._counts["used"] > 0 and vd._counts["fallback"] * 10 <= vd._counts["used"], dict(vd._counts)
        print("    lag3: steps=%d accepted=%d used=%d fallback=%d" % (s2, acc, vd._counts["used"],
                                                                       vd._counts["fallback"]))


@test
def test_device_merge_offsets():
    g = GAMMA
    Wn = vd.width(g)
    W = list(range(100, 100 + Wn))
    plan = torch.zeros((3, 2 + Wn), dtype=torch.int64)
    for i in range(3):
        plan[i, 0] = 1
        plan[i, 1] = 50
        plan[i, 2:] = torch.tensor(W)
    # row0: T = 50 (o=0) anchor W[0]; row1: T = 53 (o=3) anchor W[3]; row2: o=3 but wrong anchor
    true_len = torch.tensor([50, 53, 53], dtype=torch.int32)
    anchors = torch.tensor([W[0], W[3], 7], dtype=torch.int64)
    drafts = torch.full((3, g), JUNK, dtype=torch.int64)
    conf = torch.zeros((3, g))
    ok = vd.device_merge(plan, true_len, anchors, drafts, g, confidence=conf)
    assert ok.tolist() == [True, True, False]
    assert drafts[0].tolist() == W[1:1 + g]
    assert drafts[1].tolist() == W[4:4 + g]
    assert drafts[2].tolist() == [JUNK] * g
    assert conf[0].tolist() == [1.0] * g and conf[2].tolist() == [0.0] * g
    # beyond omax: not merged
    true_len = torch.tensor([50 + vd.omax(g) + 1] * 3, dtype=torch.int32)
    drafts = torch.full((3, g), JUNK, dtype=torch.int64)
    ok = vd.device_merge(plan, true_len, anchors, drafts, g)
    assert not ok.any()


@test
def test_non_greedy_batch_untouched():
    with Env(), NonGreedy("0"):
        runner, sim = make(temp=0.7)
        prompt, reply = rnd(64, 5), rnd(200, 6)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        sim.add("b", prompt, reply)
        sim.accepted.clear()
        sim.run(["b"])
        assert sum(sim.accepted) == 0
        assert vd._counts["steps"] == 0 and vd._state["engine"] is None, dict(vd._counts)


@test
def test_non_greedy_batch_takes_copies():
    with Env():
        runner, sim = make(temp=0.7)
        prompt, reply = rnd(64, 5), rnd(200, 6)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        sim.add("b", prompt, reply)
        sim.accepted.clear()
        sim.run(["b"])
        assert sum(sim.accepted) > 0 and vd._counts["steps"] > 0, dict(vd._counts)


@test
def test_probabilistic_drafter_keeps_sampled_batches_off():
    with Env():
        runner, sim = make(temp=0.7)
        runner.speculator.draft_logits = object()  # a cached q: a copied id would not match it
        prompt, reply = rnd(64, 5), rnd(200, 6)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        sim.add("b", prompt, reply)
        sim.accepted.clear()
        sim.run(["b"])
        assert sum(sim.accepted) == 0, dict(vd._counts)


@test
def test_mixed_batch_one_sampled_request_untouched():
    with Env(), NonGreedy("0"):
        runner, sim = make()
        prompt, reply = rnd(64, 7), rnd(300, 8)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        sim.add("b", prompt, reply)
        sim.add("c", rnd(64, 9), rnd(300, 10))
        runner.sampler.sampling_states.temperature.np[runner.req_states.req_id_to_index["c"]] = 0.8
        sim.accepted.clear()
        before = vd._counts["steps"]
        sim.run(["b", "c"])
        assert sum(sim.accepted) == 0 and vd._counts["steps"] == before, dict(vd._counts)


@test
def test_pause_file():
    with Env() as env:
        runner, sim = make()
        prompt, reply = rnd(64, 11), rnd(300, 12)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        d = vd._state["engine"].record.dir
        open(os.path.join(d, "paused"), "w").close()
        vd._state["pause_t"] = 0.0
        sim.accepted.clear()
        sim.add("b", prompt, reply)
        sim.run(["b"])
        assert sum(sim.accepted) == 0, sum(sim.accepted)
        assert vd._state["paused"] is True
        os.remove(os.path.join(d, "paused"))
        vd._state["pause_t"] = 0.0
        sim.accepted.clear()
        sim.add("c", prompt, reply)
        sim.run(["c"])
        assert sum(sim.accepted) > 100, sum(sim.accepted)


@test
def test_exception_does_not_escape():
    with Env():
        runner, sim = make()
        prompt, reply = rnd(64, 13), rnd(200, 14)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        eng = vd._state["engine"]

        def boom(*a, **k):
            raise RuntimeError("boom")

        eng.propose = boom
        sim.add("b", prompt, reply)
        sim.accepted.clear()
        sim.run(["b"])
        assert sim.out["b"] == reply.tolist()
        assert sum(sim.accepted) == 0
        assert vd._counts["errors"] > 0
        # and a broken runner state (attribute missing) is caught by the wrapper itself
        vd._state["runner"] = types.SimpleNamespace(sampler=None)
        out = sim.step(["b"], verify=False) if False else None  # noqa: F841
        b = Batch([], runner.req_states)
        w = vd._make_propose(FakeSpeculator.propose)
        w(runner.speculator, b, None, None, None, None, None, None, runner.req_states.last_sampled_tokens,
          None, None, None)


@test
def test_dummy_run_untouched():
    with Env():
        runner, sim = make()
        prompt, reply = rnd(64, 15), rnd(200, 16)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        steps = vd._counts["steps"]
        sim.add("b", prompt, reply)
        out = sim.step(["b"], verify=False, dummy=True)
        assert out.tolist() == [[JUNK] * GAMMA] and vd._counts["steps"] == steps


@test
def test_tp_plan_identical_on_all_ranks():
    """rank0 builds and broadcasts the plan; rank1 (no engine, no record) merges the same drafts."""
    class Group:
        saved = None

        def __init__(self, rank):
            self.rank_in_group = rank
            self.world_size = 2

        def broadcast(self, t, src=0):
            if self.rank_in_group == 0:
                Group.saved = t.clone()
            else:
                t.copy_(Group.saved)
            return t

    with Env():
        runner, sim = make()
        prompt, reply = rnd(64, 17), rnd(300, 18)
        sim.add("a", prompt, reply)
        sim.run(["a"])
        flush()
        sim.add("b", prompt, reply)
        sim.step(["b"], verify=False)
        for _ in range(3):
            sim.step(["b"])
        rs = runner.req_states
        sim.verify("b")
        b = Batch(["b"], rs)
        w = vd._make_propose(FakeSpeculator.propose)
        vd._state["group"] = Group(0)
        out0 = w(runner.speculator, b, None, None, None, None, None, None, rs.last_sampled_tokens, None, None,
                 None).clone()
        assert Group.saved is not None and int(Group.saved[0, 0]) == 1, Group.saved
        eng = vd._state["engine"]
        vd._state["engine"] = None  # rank1 has no engine at all
        vd._state["group"] = Group(1)
        pend = len(vd._state["pending"])
        out1 = w(runner.speculator, b, None, None, None, None, None, None, rs.last_sampled_tokens, None, None,
                 None).clone()
        vd._state["engine"] = eng
        assert out0.tolist() == out1.tolist(), (out0, out1)
        assert out0[0].tolist() != [JUNK] * GAMMA
        assert len(vd._state["pending"]) == pend, "rank1 must not snapshot"


@test
def test_install_hook_on_fake_modules():
    name_s = "xy_fake_vllm_dflash_speculator"
    name_r = "xy_fake_vllm_model_runner"
    ms = types.ModuleType(name_s)
    ms.DFlashSpeculator = type("DFlashSpeculator", (), {"propose": FakeSpeculator.propose})
    mr = types.ModuleType(name_r)

    class GPUModelRunner:
        def __init__(self):
            self.speculator = object()

        def finish_requests(self, scheduler_output):
            return "orig"

    mr.GPUModelRunner = GPUModelRunner
    sys.modules[name_s] = ms
    sys.modules[name_r] = mr
    try:
        vd.reset_state()
        vd.install({name_s: vd._patch_speculator_module, name_r: vd._patch_runner_module})
        assert ms.DFlashSpeculator.propose.__recall_wrapped__
        assert tuple(inspect.signature(ms.DFlashSpeculator.propose).parameters) == vd.PROPOSE_PARAMS
        r = mr.GPUModelRunner()
        assert vd._state["runner"] is r
        assert r.finish_requests(types.SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set())) == "orig"
        vd.install({name_s: vd._patch_speculator_module})  # idempotent
        assert ms.DFlashSpeculator.propose.__wrapped__ is FakeSpeculator.propose
    finally:
        sys.modules.pop(name_s, None)
        sys.modules.pop(name_r, None)
        vd.reset_state()


@test
def test_real_vllm_classes():
    try:
        import vllm.v1.worker.gpu.model_runner as mr
        import vllm.v1.worker.gpu.spec_decode.dflash.speculator as ms
        import vllm.v1.worker.gpu.spec_decode.dspark.speculator as mds
    except Exception as e:  # noqa: BLE001
        return "SKIP real vLLM V2 runner not importable here (%s)" % (type(e).__name__,)
    orig = ms.DFlashSpeculator.propose
    orig_names = tuple(inspect.signature(orig).parameters)
    assert orig_names == vd.PROPOSE_PARAMS, orig_names
    vd.reset_state()
    vd.install()
    try:
        p = ms.DFlashSpeculator.propose
        assert getattr(p, "__recall_wrapped__", False), "DFlashSpeculator.propose not wrapped"
        assert tuple(inspect.signature(p).parameters) == orig_names
        assert "propose" not in mds.DSparkSpeculator.__dict__, "DSpark overrides propose: hook would miss it"
        assert mds.DSparkSpeculator.propose is p
        assert getattr(mr.GPUModelRunner.__init__, "__recall_wrapped__", False)
        assert mr.GPUModelRunner.finish_requests.__wrapped__ is not None
        print("    real vLLM: DFlashSpeculator.propose wrapped (%d params), DSparkSpeculator inherits it, "
              "GPUModelRunner patched" % len(orig_names))
    finally:
        ms.DFlashSpeculator.propose = orig


def main():
    if torch is None:
        print("SKIP all: torch is not installed")
        return
    ok = fail = skip = 0
    for fn in RESULTS:
        try:
            res = fn()
            if isinstance(res, str) and res.startswith("SKIP"):
                skip += 1
                print("SKIP %s: %s" % (fn.__name__, res))
            else:
                ok += 1
                print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("torch %s, numpy %s, python %s" % (torch.__version__, np.__version__, sys.version.split()[0]))
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()

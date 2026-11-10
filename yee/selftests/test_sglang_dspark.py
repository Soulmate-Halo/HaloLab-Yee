"""CPU-only tests for the SGLang integration (needs torch; run with CUDA_VISIBLE_DEVICES= so no GPU is
touched). Run: python tests/test_recall_patch.py

Simulates SGLang's overlap schedule (the CPU processes step k-1 after launching step k), a greedy target
that follows a fixed reply, a DSpark-like drafter that is right for a random prefix, and checks that
  * replies are exactly the target's, with and without copies;
  * the CPU's alignment test equals the GPU's for every (step, row);
  * rows without an aligned copy keep their drafts; the epilogue buffer always holds what was verified;
  * a repeated request is copied (about gamma+1 ids per step);
  * the import hook stacks on top of a Mia-style finder.
"""
import collections
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU tensors only; never create a CUDA context
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
from array import array  # noqa: E402

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # no torch on this machine: the whole file is skipped
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

if torch is not None:
    from yee import core as rc  # noqa: E402
    from yee.integrations import sglang_dspark as rp  # noqa: E402

GAMMA = 5
EOS = 1
IDENT = {"tokenizer_digest": "c" * 64, "vocab_size": 129280, "model": "t"}
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn
test.__test__ = False  # decorator, not a test (pytest)


class Req:
    def __init__(self, rid, prompt, reply):
        self.rid = rid
        self.origin_input_ids = array("q", prompt)
        self.output_ids = array("q")
        self.full = list(prompt) + list(reply)  # the greedy target's sequence (reply ends with EOS)
        self.finished_reason = None
        self.is_retracted = False
        self.cached_tokens = 0
        self.seq = len(prompt)  # GPU: anchor index (ids before it are in the KV cache)
        self.gpu_done = False

    def finished(self):
        return self.finished_reason is not None


class FinishToken:
    pass


def make_reply(rng, n):
    return [int(x) for x in rng.integers(10, 50000, size=n)] + [EOS]


class Sim:
    """Overlap-scheduled DSpark decode with the patch's three pieces wired in."""

    def __init__(self, engine, seed=0):
        self.engine = engine
        self.rng = random.Random(seed)
        self.pending = collections.deque()  # results not yet processed by the CPU: (kind, reqs, commits, gpu_ok)
        self.batch = []
        self.mismatch = 0
        self.checked = 0
        self.used_rows = 0
        self.decode_rows = 0
        self.commit_hist = []
        self.untouched_bad = 0
        self.epi_bad = 0
        self.epi = torch.zeros(64 * GAMMA, dtype=torch.int64)

    def admit(self, reqs):
        # a prefill batch is this iteration's launch: the GPU commits each first reply id, and the CPU
        # learns them when it processes this result one iteration later
        self.pending.append(("prefill", list(reqs), [[r.full[r.seq]] for r in reqs], None))
        self.batch.extend(reqs)

    def _drafts(self, req):
        good = self.rng.randint(0, GAMMA)
        d = []
        for i in range(1, GAMMA + 1):
            j = req.seq + i
            if i <= good and j < len(req.full):
                d.append(req.full[j])
            else:
                d.append(self.rng.randint(50001, 60000))
        return d

    def launch(self):
        reqs = [r for r in self.batch if not r.finished()]
        if not reqs:
            return
        plan, entries = rp.build_plan(self.engine, reqs, GAMMA)
        for e in entries:
            rp._fifo.setdefault(e[0], collections.deque()).append(e)
        drafts = torch.tensor([self._drafts(r) for r in reqs], dtype=torch.int64)
        orig = drafts.clone()
        seq = torch.tensor([r.seq for r in reqs], dtype=torch.int64)
        anchors = torch.tensor([r.full[r.seq] for r in reqs], dtype=torch.int64)
        conf = torch.full((len(reqs), GAMMA), 0.3)
        ok = rp.device_merge(torch.from_numpy(plan), seq, anchors, drafts, conf, GAMMA, epilogue_buf=self.epi)
        bs = len(reqs)
        if not torch.equal(self.epi[: bs * GAMMA].view(bs, GAMMA), drafts):
            self.epi_bad += 1
        if not torch.equal(drafts[~ok], orig[~ok]):
            self.untouched_bad += 1
        if ok.any() and not bool((conf[ok] == 1.0).all()):
            self.untouched_bad += 1
        commits = []
        for i, r in enumerate(reqs):
            if r.gpu_done:  # overshoot tick of a finished row: SGLang skips it when processing
                commits.append([])
                continue
            c = 0
            while c < GAMMA and r.seq + c + 1 < len(r.full) and int(drafts[i, c]) == r.full[r.seq + c + 1]:
                c += 1
            toks = r.full[r.seq + 1:r.seq + c + 2]
            if EOS in toks:
                toks = toks[:toks.index(EOS) + 1]
                r.gpu_done = True
            r.seq += len(toks)
            commits.append(toks)
        self.pending.append(("decode", reqs, commits, ok.tolist()))

    def process_one(self):
        if not self.pending:
            return
        kind, reqs, commits, gpu_ok = self.pending.popleft()
        if kind == "prefill":
            for r, toks in zip(reqs, commits):
                r.output_ids.extend(toks)
            return
        before = [len(r.origin_input_ids) + len(r.output_ids) for r in reqs]
        for r, toks in zip(reqs, commits):  # what process_batch_result_decode does
            if r.finished() or not toks:
                continue
            r.output_ids.extend(toks)
            if r.output_ids[-1] == EOS:
                r.finished_reason = FinishToken()
        for i, (r, Lb) in enumerate(zip(reqs, before)):
            dq = rp._fifo.get(r.rid)
            entry = dq[0] if dq else None
            rp.after_step(self.engine, r, Lb, GAMMA)
            if entry is None:
                continue
            cpu_used = rp._counts["used"]
            if entry[1]:
                self.checked += 1
            self.decode_rows += 1
            # recompute the CPU decision exactly as account() does, compare with the GPU's
            rid, flag, L, W, real = entry
            o = Lb - L
            n0 = len(r.origin_input_ids)
            tok = r.origin_input_ids[Lb - 1] if Lb - 1 < n0 else r.output_ids[Lb - 1 - n0]
            cpu_ok = bool(flag) and 0 <= o <= rp.O_MAX and W[o] == tok
            if cpu_ok != bool(gpu_ok[i]):
                self.mismatch += 1
            if cpu_ok:
                self.used_rows += 1
            if commits[i]:
                self.commit_hist.append(len(commits[i]))
            _ = cpu_used
        self.batch = [r for r in self.batch if not r.finished()]

    def run(self, schedule, max_iter=10000):
        """schedule: {iteration: [reqs to admit]}. Like SGLang's overlap loop, each iteration launches one
        batch (a prefill batch when requests arrive, else the decode batch) and then processes the result
        of the previous iteration's batch, so the CPU is 0-1 steps behind the GPU."""
        it = 0
        while it < max_iter:
            adm = schedule.get(it, [])
            if adm:
                self.admit(adm)
            else:
                self.launch()
            self.process_one()
            it += 1
            if not self.batch and not self.pending and it > max(schedule):
                break
        while self.pending:
            self.process_one()
        return it


def _engine(d):
    rec = rc.RecallRecord(d, IDENT, chunk_ids=1 << 14)
    return rc.RecallEngine(rec, gamma=GAMMA, async_add=False)


@test
def overlap_exact_and_aligned():
    d = tempfile.mkdtemp(prefix="rp_t_")
    try:
        rng = np.random.default_rng(1)
        rp._fifo.clear()
        rp._counts.clear()
        eng = _engine(os.path.join(d, "rec"))
        prompts = [[int(x) for x in rng.integers(10, 50000, size=n)] for n in (40, 90, 300)]
        replies = [make_reply(rng, n) for n in (180, 260, 120)]
        # pass 1: requests join at different iterations, finish at different times
        p1 = [Req("a%d" % i, p, r) for i, (p, r) in enumerate(zip(prompts, replies))]
        s1 = Sim(eng, seed=2)
        it1 = s1.run({0: [p1[0]], 3: [p1[1]], 7: [p1[2]]})
        for r in p1:
            assert list(r.output_ids) == r.full[len(r.origin_input_ids):], r.rid
        assert s1.mismatch == 0, s1.mismatch
        assert s1.untouched_bad == 0 and s1.epi_bad == 0, (s1.untouched_bad, s1.epi_bad)
        # pass 2: the same prompts again, the record holds pass 1's replies
        p2 = [Req("b%d" % i, p, r) for i, (p, r) in enumerate(zip(prompts, replies))]
        s2 = Sim(eng, seed=3)
        it2 = s2.run({0: [p2[0]], 3: [p2[1]], 7: [p2[2]]})
        for r in p2:
            assert list(r.output_ids) == r.full[len(r.origin_input_ids):], r.rid
        assert s2.mismatch == 0, s2.mismatch
        assert s2.untouched_bad == 0 and s2.epi_bad == 0
        frac = s2.used_rows / max(1, s2.decode_rows)
        mean_commit2 = float(np.mean(s2.commit_hist))
        mean_commit1 = float(np.mean(s1.commit_hist))
        print("    pass1: iters=%d rows=%d used=%d mean_commit=%.2f | pass2: iters=%d rows=%d used=%d (%.0f%%) "
              "mean_commit=%.2f | checked=%d+%d" % (it1, s1.decode_rows, s1.used_rows, mean_commit1, it2,
                                                    s2.decode_rows, s2.used_rows, 100 * frac, mean_commit2,
                                                    s1.checked, s2.checked))
        assert s1.used_rows <= 2, s1.used_rows  # nothing to copy the first time
        assert frac > 0.8, frac
        assert mean_commit2 > 5.0, mean_commit2
        assert it2 < it1 * 0.75, (it1, it2)
        assert not rp._fifo, rp._fifo  # every finished request cleaned up
        eng.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test
def merge_rules():
    plan = torch.zeros((4, 2 + GAMMA + rp.PAD), dtype=torch.int64)
    W = list(range(100, 112))
    for i in range(4):
        plan[i, 0] = 1
        plan[i, 1] = 50
        plan[i, 2:] = torch.tensor(W)
    plan[3, 0] = 0  # no copy offered
    seq = torch.tensor([49, 52, 55, 49])  # o = 0, 3, 6(ok if anchor matches), 0
    anchors = torch.tensor([100, 103, 999, 100])  # row 2: anchor mismatch -> keep drafts
    drafts = torch.full((4, GAMMA), 7, dtype=torch.int64)
    conf = torch.full((4, GAMMA), 0.2)
    ok = rp.device_merge(plan, seq, anchors, drafts, conf, GAMMA)
    assert ok.tolist() == [True, True, False, False], ok
    assert drafts[0].tolist() == [101, 102, 103, 104, 105]
    assert drafts[1].tolist() == [104, 105, 106, 107, 108]
    assert drafts[2].tolist() == [7] * 5 and drafts[3].tolist() == [7] * 5
    assert conf[0].tolist() == [1.0] * 5 and abs(conf[2, 0].item() - 0.2) < 1e-6
    seq = torch.tensor([56, 42, 49, 49])  # o = 7 and o = -7: out of range
    drafts = torch.full((4, GAMMA), 7, dtype=torch.int64)
    ok = rp.device_merge(plan, seq, torch.tensor([0, 0, 100, 100]), drafts, None, GAMMA)
    assert ok.tolist() == [False, False, True, False]


@test
def on_propose_wiring():
    """_on_propose with a fake TP group / worker / batch: rank0 fills FIFO, plan reaches the merge."""
    d = tempfile.mkdtemp(prefix="rp_w_")
    try:
        rp._fifo.clear()
        rp._counts.clear()
        eng = _engine(os.path.join(d, "rec"))
        rng = np.random.default_rng(5)
        prompt = [int(x) for x in rng.integers(10, 50000, size=60)]
        reply = make_reply(rng, 80)
        eng.begin_stream("old", prompt)
        eng.end_stream("old", reply)
        req = Req("new", prompt, reply)
        req.output_ids.extend(reply[:1])  # first id known on the CPU (lag 0)

        class G:
            rank_in_group = 0

            def broadcast(self, t, src=0):
                assert src == 0
                return t

        class B:
            reqs = [req]
            seq_lens = torch.tensor([len(prompt)])  # anchor = first reply id
            has_grammar = False

        class SI:
            is_all_greedy = True

        class DB:
            draft_tokens = torch.full((1, GAMMA), 7, dtype=torch.int64)

        class Out:
            draft_block = DB()
            draft_block_ids = torch.tensor([[reply[0]] + [0] * GAMMA])
            confidence = torch.full((1, GAMMA), 0.1)

        class Epi:
            draft_tokens_buf = torch.zeros(64 * GAMMA, dtype=torch.int64)

        class Wk:
            _verify_epilogue = Epi()

        rp._state.update(engine=eng, engine_failed=False, group=G(), paused=False, pause_t=1e18)
        real_pinned = rp._pinned
        rp._pinned = lambda width: torch.zeros((rp.MAX_BS, width), dtype=torch.int64)
        try:
            kw = {"batch": B(), "bs": 1, "sampling_info": SI()}
            out = Out()
            assert rp._participates(kw, out)
            rp._on_propose(Wk(), kw, out)
        finally:
            rp._pinned = real_pinned
            rp._state.update(engine=None, group=None, pause_t=0.0)
        assert out.draft_block.draft_tokens[0].tolist() == reply[1:1 + GAMMA], out.draft_block.draft_tokens
        assert Wk._verify_epilogue.draft_tokens_buf[:GAMMA].tolist() == reply[1:1 + GAMMA]
        assert float(out.confidence.min()) == 1.0
        assert len(rp._fifo["new"]) == 1 and rp._fifo["new"][0][1] == 1
        kw_s = {"batch": B(), "bs": 1, "sampling_info": type("S", (), {"is_all_greedy": False})()}
        assert not rp._participates(kw_s, Out())
        eng.close()
    finally:
        rp._fifo.clear()
        shutil.rmtree(d, ignore_errors=True)


@test
def finder_stacks_on_mia_style_finder():
    import importlib.abc
    import importlib.machinery

    d = tempfile.mkdtemp(prefix="rp_f_")
    try:
        pkg = os.path.join(d, "fakepkg_r581")
        os.makedirs(pkg)
        open(os.path.join(pkg, "__init__.py"), "w").close()
        with open(os.path.join(pkg, "target.py"), "w") as f:
            f.write("class K:\n    pass\n")
        sys.path.insert(0, d)

        class MiaLoader(importlib.abc.Loader):
            def __init__(self, inner):
                self.inner = inner

            def create_module(self, spec):
                return self.inner.create_module(spec)

            def exec_module(self, module):
                self.inner.exec_module(module)
                module.mia = True

        class MiaFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != "fakepkg_r581.target":
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec is not None:
                    spec.loader = MiaLoader(spec.loader)
                return spec

        sys.meta_path.insert(0, MiaFinder())
        rp.install(targets={"fakepkg_r581.target": lambda m: setattr(m, "recall", True)})
        import fakepkg_r581.target as t  # noqa: E402

        assert getattr(t, "mia", False) and getattr(t, "recall", False), vars(t).keys()
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ not in ("MiaFinder", "_Finder")]
        if d in sys.path:
            sys.path.remove(d)
        shutil.rmtree(d, ignore_errors=True)


def main():
    if torch is None:
        print("SKIP all: torch is not installed on this machine")
        print("ALL PASS n=0 (skipped)")
        return
    ok = fail = 0
    for fn in TESTS:
        try:
            fn()
            ok += 1
            print("PASS %s" % fn.__name__)
        except Exception:
            fail += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("torch %s numpy %s python %s" % (torch.__version__, np.__version__, sys.version.split()[0]))
    if fail:
        print("FAILED %d / %d" % (fail, ok + fail))
        sys.exit(1)
    print("ALL PASS n=%d" % ok)


if __name__ == "__main__":
    main()

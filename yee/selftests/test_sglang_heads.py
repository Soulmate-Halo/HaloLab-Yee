"""Plain-assert tests for the SGLang EAGLE / DFLASH / NGRAM adapters and the registry. Run: python tests/test_sglang_heads.py

CPU simulations with fake modules that keep the real call shapes (ysy37 dsv41-head sources):
  * EAGLE: ``build_eagle_verify_input(batch, draft_input, parent_list, top_scores_index, draft_tokens, draft_probs, *,
    target_worker, topk, num_steps, num_draft_tokens, tree_mask_mode, device)``; the anchor (bonus) is at
    position ``seq_lens``; drafts [bs, num_steps].
  * DFLASH: ``DFlashWorkerV2.forward_batch_generation`` fills ``[bonus, drafts...]`` into one buffer and builds the
    module-level ``DFlashVerifyInput(draft_token=buf.reshape(-1), draft_token_num=n, ...)``; accept reads the buffer.
  * NGRAM: ``NGRAMWorker._prepare_draft_tokens(batch) -> (req_drafts, mask)`` numpy, root first, n*n mask per row,
    overlap-spliced ids in ``prev_token_ids`` / ``prev_accept_lens``.
The fake target is greedy with a fixed reply per prompt; verify accepts the longest matching draft prefix + 1 bonus,
and the scheduler's result processor runs one step behind (overlap) when ``lag=1``.
"""
from __future__ import annotations

import collections
import os
import shutil
import sys
import tempfile
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yee.integrations import registry  # noqa: E402
from yee.integrations import sglang_common as sc  # noqa: E402
from yee.integrations import sglang_dflash as sdf  # noqa: E402
from yee.integrations import sglang_dspark as sd  # noqa: E402
from yee.integrations import sglang_eagle as se  # noqa: E402
from yee.integrations import sglang_ngram as sn  # noqa: E402

RESULTS = []
JUNK = 3
G = 5


def test(fn):
    RESULTS.append(fn)
    return fn


def rnd(n, seed, lo=10, hi=120000):
    return [int(x) for x in np.random.default_rng(seed).integers(lo, hi, size=n)]


# ------------------------------------------------------------------------------------------- fakes
class FinishLength:
    pass


class Req:
    def __init__(self, rid, prompt, reply_len):
        self.rid = rid
        self.origin_input_ids = list(prompt)
        self.output_ids = []
        self.cached_tokens = 0
        self.reply_len = reply_len
        self.finished_reason = None
        self.is_retracted = False

    def finished(self):
        return len(self.output_ids) >= self.reply_len


class Mode:
    def __init__(self, idle=False):
        self.idle = idle

    def is_idle(self):
        return self.idle


class SpecAlg:
    def is_none(self):
        return False


class Batch:
    def __init__(self, reqs, seq_lens, greedy=True, grammar=False):
        self.reqs = reqs
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
        self.sampling_info = types.SimpleNamespace(is_all_greedy=greedy)
        self.has_grammar = grammar
        self.forward_mode = Mode()
        self.spec_algorithm = SpecAlg()


def make_processor():
    class SchedulerBatchResultProcessor:
        def process_batch_result_decode(self, batch, result):
            for req, toks in zip(batch.reqs, result):
                req.output_ids.extend(toks)
                if req.finished():
                    req.finished_reason = FinishLength()

    mod = types.ModuleType("fake_processor")
    mod.SchedulerBatchResultProcessor = SchedulerBatchResultProcessor
    sd._patch_processor_module(mod)
    return SchedulerBatchResultProcessor()


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
    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="ra_heads_")
        self.keys = {"RECALL_ACCEL": "1", "RECALL_ACCEL_DIR": os.path.join(self.d, "rec"),
                     "RECALL_ACCEL_GIB": "0.01", "RECALL_ACCEL_BACKEND": "memory",
                     "RECALL_ACCEL_TOKENIZER_DIR": self.d}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ.update(self.keys)
        reset()
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reset()
        shutil.rmtree(self.d, ignore_errors=True)


def reset():
    sd._state.update(engine=None, engine_failed=False, group=None, pinned=None, slot=0, worker=None, paused=False,
                     pause_t=0.0, log_t=0.0)
    sd._fifo.clear()
    sd._counts.clear()
    sc._rings.clear()


def drive(draft_fn, jobs, g, lag, max_steps=5000):
    """jobs: [(rid, prompt, reply)]. draft_fn(batch, gpu_rows) -> np drafts [bs, g]. Returns steps."""
    proc = make_processor()
    reqs = [Req(rid, p, len(r)) for rid, p, r in jobs]
    gpu = [list(p) for _, p, _ in jobs]
    replies = [list(r) for _, _, r in jobs]
    prompts = [len(p) for _, p, _ in jobs]
    pending = collections.deque()
    steps = 0
    while steps < max_steps:
        act = [i for i in range(len(jobs)) if len(gpu[i]) - prompts[i] < len(replies[i])]
        if not act:
            break
        batch = Batch([reqs[i] for i in act], [len(gpu[i]) - 1 for i in act])
        drafts = draft_fn(batch, [gpu[i] for i in act], pending)
        commits = []
        for row, i in enumerate(act):
            pos = len(gpu[i]) - prompts[i]
            ref = replies[i]
            acc = 0
            while acc < g and pos + acc < len(ref) and int(drafts[row][acc]) == ref[pos + acc]:
                acc += 1
            new = ref[pos:pos + acc + 1]
            gpu[i].extend(new)
            commits.append(new)
        steps += 1
        pending.append((batch, commits))
        while len(pending) > lag:
            b, c = pending.popleft()
            proc.process_batch_result_decode(b, c)
    while pending:
        b, c = pending.popleft()
        proc.process_batch_result_decode(b, c)
    for i, r in enumerate(reqs):
        assert r.output_ids == replies[i], (i, len(r.output_ids), len(replies[i]))
    return steps


# ------------------------------------------------------------------------------------------- EAGLE
def eagle_build():
    def build_eagle_verify_input(batch, draft_input, parent_list, top_scores_index, draft_tokens, draft_probs, *,
                                 target_worker, topk, num_steps, num_draft_tokens, tree_mask_mode, device):
        return torch.cat((draft_input.bonus_tokens.unsqueeze(1), draft_tokens), dim=1)

    return se.wrap_build(build_eagle_verify_input)


def eagle_fn(g, topk=1, probs=None):
    build = eagle_build()

    def fn(batch, rows, pending):
        bs = len(rows)
        bonus = torch.tensor([r[-1] for r in rows], dtype=torch.int64)
        dt = torch.full((bs, g), JUNK, dtype=torch.int64)
        di = types.SimpleNamespace(bonus_tokens=bonus)
        out = build(batch, di, None, None, dt, probs, target_worker=None, topk=topk, num_steps=g,
                    num_draft_tokens=g + 1, tree_mask_mode=None, device="cpu")
        assert torch.equal(out[:, 0], bonus)
        return out[:, 1:].numpy()

    return fn


def two_passes(fn_factory, g, lag, n_reply=300, seed=1):
    prompt = rnd(200, seed)
    reply = rnd(n_reply, seed + 100)
    s1 = drive(fn_factory(), [("a", prompt, reply)], g, lag)
    s2 = drive(fn_factory(), [("b", prompt, reply)], g, lag)
    return s1, s2


@test
def test_eagle_second_pass_copies():
    with Env():
        s1, s2 = two_passes(lambda: eagle_fn(G), G, lag=0)
        assert s1 == 300, s1
        assert s2 < 300 * 0.45, (s1, s2, dict(sd._counts))
        assert sd._counts["used"] > 0, dict(sd._counts)
        print("    eagle lag0: %d -> %d steps" % (s1, s2))


@test
def test_eagle_lag_one_step():
    with Env():
        s1, s2 = two_passes(lambda: eagle_fn(G), G, lag=1, seed=31)
        assert s1 == 300 and s2 < 300 * 0.5, (s1, s2, dict(sd._counts))
        assert sd._counts["fallback"] * 10 <= sd._counts["used"] + 10, dict(sd._counts)
        print("    eagle lag1: %d -> %d steps used=%d fallback=%d" % (s1, s2, sd._counts["used"],
                                                                     sd._counts["fallback"]))


@test
def test_eagle_two_requests_batch():
    with Env():
        p1, p2 = rnd(150, 5), rnd(150, 6)
        r1, r2 = rnd(200, 7), rnd(200, 8)
        drive(eagle_fn(G), [("a", p1, r1), ("b", p2, r2)], G, lag=1)
        s2 = drive(eagle_fn(G), [("c", p1, r1), ("d", p2, r2)], G, lag=1)
        assert s2 < 200 * 0.5, s2


@test
def test_eagle_off_cases():
    with Env():
        prompt, reply = rnd(200, 11), rnd(120, 12)
        drive(eagle_fn(G), [("a", prompt, reply)], G, lag=0)
        # tree (topk>1), rejection sampling (draft_probs), non-greedy: drafts untouched
        assert drive(eagle_fn(G, topk=4), [("b", prompt, reply)], G, lag=0) == 120
        assert drive(eagle_fn(G, probs=torch.zeros(1)), [("c", prompt, reply)], G, lag=0) == 120
        build = eagle_build()

        def sampled_build():
            b = Batch([Req("x", prompt, 10)], [len(prompt) - 1], greedy=False)
            dt = torch.full((1, G), JUNK, dtype=torch.int64)
            return build(b, types.SimpleNamespace(bonus_tokens=torch.tensor([prompt[-1]])), None, None, dt, None,
                         target_worker=None, topk=1, num_steps=G, num_draft_tokens=G + 1, tree_mask_mode=None,
                         device="cpu")

        with NonGreedy("0"):  # sampled batches off when the switch is off
            assert bool((sampled_build()[:, 1:] == JUNK).all())
        # default: a sampled batch takes the copy (target-only verify keeps p)
        out = sampled_build()
        assert not bool((out[:, 1:] == JUNK).all()), out
        assert out[0, 1:].tolist() == [int(x) for x in reply[:G]], out


@test
def test_eagle_errors_contained():
    with Env():
        orig = sd.build_plan

        def boom(*a, **k):
            raise RuntimeError("boom")

        sd.build_plan = boom
        try:
            prompt, reply = rnd(100, 21), rnd(50, 22)
            assert drive(eagle_fn(G), [("a", prompt, reply)], G, lag=1) == 50
            assert sd._counts["plan_errors"] > 0
        finally:
            sd.build_plan = orig


@test
def test_eagle_patch_targets():
    calls = []

    def build_eagle_verify_input(batch, draft_input, parent_list, top_scores_index, draft_tokens, draft_probs, *,
                                 target_worker, topk, num_steps, num_draft_tokens, tree_mask_mode, device):
        calls.append(1)

    common = types.ModuleType(se.COMMON)
    common.build_eagle_verify_input = build_eagle_verify_input
    caller = types.ModuleType(se.CALLERS[0])
    caller.build_eagle_verify_input = build_eagle_verify_input
    se._patch_common(common)
    se._patch_caller(caller)
    w = common.build_eagle_verify_input
    assert getattr(w, "__recall_wrapped__", False) and caller.build_eagle_verify_input.__recall_wrapped__
    se._patch_caller(caller)  # idempotent
    assert caller.build_eagle_verify_input.__wrapped__ is build_eagle_verify_input
    import inspect

    assert tuple(inspect.signature(w).parameters) == se.BUILD_PARAMS


# ------------------------------------------------------------------------------------------- DFLASH
DFLASH_SRC = '''
import torch

class DFlashVerifyInput:
    def __init__(self, draft_token=None, positions=None, draft_token_num=0, custom_mask=None, capture_hidden_mode=None):
        self.draft_token = draft_token
        self.draft_token_num = draft_token_num

def _is_all_greedy(si):
    return bool(getattr(si, "is_all_greedy", False))

class DFlashWorkerV2:
    def __init__(self, block):
        self.block_size = block
        self._selector_sample = None
        self._buf = torch.zeros((64, block), dtype=torch.int64)

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None):
        bs = len(batch.reqs)
        draft_tokens = self._buf[:bs]
        draft_tokens[:, 0].copy_(batch.bonus)
        draft_tokens[:, 1:].fill_(%d)
        verify_input = DFlashVerifyInput(draft_token=draft_tokens.reshape(-1), positions=None,
                                         draft_token_num=int(self.block_size))
        candidates = draft_tokens
        return candidates[:, 1:].clone()

    def _accept_block(self, **kw):
        pass
''' % JUNK


def dflash_module():
    mod = types.ModuleType(sdf.WORKER)
    exec(compile(DFLASH_SRC, "<fake dflash_worker_v2>", "exec"), mod.__dict__)
    sys.modules[sdf.WORKER] = mod
    sdf._patch_worker(mod)
    return mod


def dflash_fn(mod, g):
    w = mod.DFlashWorkerV2(g + 1)

    def fn(batch, rows, pending):
        batch.bonus = torch.tensor([r[-1] for r in rows], dtype=torch.int64)
        return w.forward_batch_generation(batch).numpy()

    return fn


@test
def test_dflash_second_pass_copies():
    with Env():
        mod = dflash_module()
        try:
            for lag in (0, 1):
                reset()
                s1, s2 = two_passes(lambda: dflash_fn(mod, 15), 15, lag=lag, seed=40 + lag)
                assert s1 == 300 and s2 < 300 * 0.3, (lag, s1, s2, dict(sd._counts))
                print("    dflash block16 lag%d: %d -> %d steps" % (lag, s1, s2))
            sdf._patch_worker(mod)  # idempotent
            assert mod.DFlashVerifyInput.__wrapped__.__name__ == "DFlashVerifyInput"
        finally:
            sys.modules.pop(sdf.WORKER, None)


@test
def test_dflash_selector_and_nongreedy_off():
    with Env():
        mod = dflash_module()
        try:
            prompt, reply = rnd(200, 51), rnd(100, 52)
            drive(dflash_fn(mod, 7), [("a", prompt, reply)], 7, lag=0)
            w = mod.DFlashWorkerV2(8)
            w._selector_sample = (1, 2)

            def fn(batch, rows, pending):
                batch.bonus = torch.tensor([r[-1] for r in rows], dtype=torch.int64)
                return w.forward_batch_generation(batch).numpy()

            assert drive(fn, [("b", prompt, reply)], 7, lag=0) == 100

            def sampled():
                b = Batch([Req("x", prompt, 5)], [len(prompt) - 1], greedy=False)
                b.bonus = torch.tensor([prompt[-1]])
                return mod.DFlashWorkerV2(8).forward_batch_generation(b)

            with NonGreedy("0"):
                assert bool((sampled() == JUNK).all())
            assert not bool((sampled() == JUNK).all())  # default: sampled batches take copies
        finally:
            sys.modules.pop(sdf.WORKER, None)


# ------------------------------------------------------------------------------------------- NGRAM
class FakeNgram:
    def __init__(self, n):
        self.draft_token_num = n
        self.device = "cpu"
        self.prev_token_ids = []
        self.prev_accept_lens = []
        self.masks = None

    def _prepare_draft_tokens(self, batch):
        bs = len(batch.reqs)
        n = self.draft_token_num
        drafts = np.full(bs * n, JUNK, dtype=np.int64)
        for i, r in enumerate(batch.reqs):
            k = self.prev_accept_lens[i] if i < len(self.prev_accept_lens) else 0
            prev = self.prev_token_ids[i * n:i * n + k]
            drafts[i * n] = (list(r.origin_input_ids) + list(r.output_ids) + list(prev))[-1]
        mask = np.zeros((bs, n, n), dtype=bool)
        mask[:, :, 0] = True
        for q in range(n):
            mask[:, q, q] = True
        return drafts, mask.reshape(-1)


def ngram_fn(g, worker=None):
    n = g + 1
    w = worker or FakeNgram(n)
    w._prepare_draft_tokens = types.MethodType(sn.wrap_prepare(FakeNgram._prepare_draft_tokens), w)

    def fn(batch, rows, pending):
        bs = len(batch.reqs)
        ids, lens = [], []
        last = pending[-1] if pending else None  # overlap: the previous step's result is not processed yet
        prev = {}
        if last is not None:
            for r, c in zip(last[0].reqs, last[1]):
                prev[r.rid] = c
        for r in batch.reqs:
            p = list(prev.get(r.rid, []))[:n]
            ids.extend(p + [0] * (n - len(p)))
            lens.append(len(p))
        w.prev_token_ids, w.prev_accept_lens = ids, lens
        drafts, mask = w._prepare_draft_tokens(batch)
        d2 = drafts.reshape(bs, n)
        m3 = mask.reshape(bs, n, n)
        tri = np.tril(np.ones((n, n), dtype=bool))
        for i in range(bs):
            assert d2[i, 0] == rows[i][-1], "root must stay the current last id"
            if (d2[i, 1:] != JUNK).any():
                assert (m3[i] == tri).all(), "a copied row must carry a chain mask"
        return d2[:, 1:]

    return fn


@test
def test_ngram_second_pass_copies():
    with Env():
        for lag in (0, 1):
            reset()
            s1, s2 = two_passes(lambda: ngram_fn(G), G, lag=lag, seed=60 + lag)
            assert s1 == 300 and s2 < 300 * 0.5, (lag, s1, s2, dict(sd._counts))
            print("    ngram n=6 lag%d: %d -> %d steps" % (lag, s1, s2))


@test
def test_ngram_row_checks():
    W = np.array([1, 50, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)  # flag, L, W0..
    assert sn.row_ok(W, [], 7, 5)
    assert sn.row_ok(W, [11, 12], 12, 5)
    assert not sn.row_ok(W, [11, 99], 99, 5)
    assert not sn.row_ok(W, [11, 12], 13, 5)
    W0 = W.copy()
    W0[0] = 0
    assert not sn.row_ok(W0, [], 7, 5)
    assert sn.participates(FakeNgram(6), Batch([Req("x", [1], 1)], [0]), np.zeros(6), np.zeros(36))
    assert not sn.participates(FakeNgram(6), Batch([Req("x", [1], 1)], [0], grammar=True), np.zeros(6), np.zeros(36))
    assert not sn.participates(FakeNgram(6), Batch([Req("x", [1], 1)], [0]), np.zeros(5), np.zeros(36))


# ------------------------------------------------------------------------------------------- registry
@test
def test_registry_choose():
    c = registry.choose
    assert c("sglang", "DSPARK")["adapter"] == "sglang_dspark"
    for alg in ("EAGLE", "EAGLE3", "NEXTN", "MTP", "STANDALONE", "UNO"):
        assert c("sglang", alg)["adapter"] == "sglang_eagle", alg
    assert c("sglang", "dflash")["adapter"] == "sglang_dflash"
    assert c("sglang", "NGRAM")["adapter"] == "sglang_ngram"
    assert c("sglang", "EAGLE", topk=4)["status"] == "no-op"
    assert c("sglang", "FROZEN_KV_MTP")["adapter"] is None
    assert c("sglang", "WHATEVER")["adapter"] is None and "not in" in c("sglang", "WHATEVER")["reason"]
    assert c("sglang", None)["adapter"] is None and "NGRAM" in c("sglang", None)["reason"]
    for m in ("dspark", "dflash", "eagle", "eagle3", "mtp", "deepseek_mtp", "dflash2"):
        assert c("vllm", m)["adapter"] == "vllm_spec", m
    assert c("vllm", "ngram")["adapter"] is None
    assert c("tensorfold", "RECALL")["status"] == "native"
    assert c("trtllm", "EAGLE")["adapter"] is None


@test
def test_registry_cmdline():
    p = registry.parse_cmdline("/opt/sglang/bin/python3 -m sglang.launch_server --model-path /m --tp 4 "
                               "--speculative-algorithm DSPARK --speculative-dspark-block-size 7")
    assert p["engine"] == "sglang" and p["algorithm"] == "DSPARK"
    p = registry.parse_cmdline("python -m sglang.launch_server --speculative-algorithm=EAGLE3 --speculative-eagle-topk 1")
    assert p["algorithm"] == "EAGLE3" and p["topk"] == 1
    p = registry.parse_cmdline(["vllm", "serve", "/m", "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}'])
    assert p["engine"] == "vllm" and p["algorithm"] == "DSPARK"
    p = registry.parse_cmdline("tensorfold serve --model x --recall-gib 0.5")
    assert p["engine"] == "tensorfold"
    p = registry.parse_cmdline("python app.py")
    assert p["engine"] is None


@test
def test_registry_install_off_and_merge():
    old = os.environ.get("RECALL_ACCEL_ADAPTER")
    os.environ["RECALL_ACCEL_ADAPTER"] = "off"
    try:
        assert registry.install_auto() == []
    finally:
        if old is None:
            os.environ.pop("RECALL_ACCEL_ADAPTER", None)
        else:
            os.environ["RECALL_ACCEL_ADAPTER"] = old
    merged = registry._merge_targets(registry.SGLANG_ALL)
    for name in (se.COMMON, sdf.WORKER, sn.WORKER, "sglang.srt.speculative.dspark_components.dspark_worker_v2",
                 se.PROCESSOR):
        assert name in merged, name
    # the processor patch is idempotent, so merging it from four adapters patches once
    mod = types.ModuleType("p")

    class SchedulerBatchResultProcessor:
        def process_batch_result_decode(self, batch, result):
            return "orig"

    mod.SchedulerBatchResultProcessor = SchedulerBatchResultProcessor
    merged[se.PROCESSOR](mod)
    f = SchedulerBatchResultProcessor.process_batch_result_decode
    merged[se.PROCESSOR](mod)
    assert SchedulerBatchResultProcessor.process_batch_result_decode is f


@test
def test_registry_check_symbols_offline():
    rows = registry.check_symbols("sglang_eagle")
    assert rows and all(ok is False for ok, _ in rows if ok is not None) or "sglang" in sys.modules, rows


def main():
    if torch is None:
        print("SKIP all: torch is not installed")
        print("ALL PASS n=0 (skipped)")
        return
    ok = fail = 0
    for fn in RESULTS:
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

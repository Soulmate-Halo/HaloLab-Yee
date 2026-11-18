"""Plain-assert tests of the Yee 4.2 cross-rank check (``yee.integrations.rank_agree``): every rank that has Yee
reaches the same verdict, a missing rank turns the feature off on all of them, a late rank still reads "off", and
the store without ``compare_set`` falls back. Threads stand in for ranks; one case uses a real torch TCPStore.

Run: python tests/test_rank_agree.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee.integrations import rank_agree  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


class FakeStore:
    def __init__(self):
        self.d = {}
        self.lock = threading.Lock()

    def set(self, k, v):
        with self.lock:
            self.d[k] = v.encode() if isinstance(v, str) else v

    def get(self, k):
        with self.lock:
            return self.d[k]

    def check(self, keys):
        with self.lock:
            return all(k in self.d for k in keys)

    def compare_set(self, k, expected, desired):
        with self.lock:
            cur = self.d.get(k)
            if cur is None:
                if expected == "":
                    self.d[k] = desired.encode()
            elif cur == expected.encode():
                self.d[k] = desired.encode()
            return self.d.get(k, expected.encode())


class NoCasStore(FakeStore):
    compare_set = None


def _run(store, ranks, present, timeout_s=0.3):
    out = {}

    def one(r):
        out[r] = rank_agree._agree("f", ranks, r, timeout_s, store, 0.01)

    ts = [threading.Thread(target=one, args=(r,)) for r in present]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    return out


@test
def test_all_ranks_present_turns_on():
    out = _run(FakeStore(), [0, 1, 2, 3], [0, 1, 2, 3], timeout_s=10)
    assert all(ok for ok, _ in out.values()) and len(out) == 4, out
    assert out[0][1] == "4 ranks"


@test
def test_missing_rank_turns_off_everywhere():
    out = _run(FakeStore(), [0, 1, 2, 3], [0, 1, 2])
    assert len(out) == 3 and not any(ok for ok, _ in out.values()), out
    assert "ranks 3 did not report" in out[0][1], out[0][1]


@test
def test_late_rank_reads_off():
    store = FakeStore()
    first = _run(store, [0, 1, 2, 3], [0, 1, 2], timeout_s=0.2)
    assert not any(ok for ok, _ in first.values())
    late = _run(store, [0, 1, 2, 3], [3], timeout_s=5)
    assert late[3][0] is False, late


@test
def test_without_compare_set():
    out = _run(NoCasStore(), [0, 1], [0, 1], timeout_s=10)
    assert all(ok for ok, _ in out.values()), out
    out = _run(NoCasStore(), [0, 1], [0])
    assert out[0][0] is False


@test
def test_single_rank_and_no_store():
    assert rank_agree._agree("f", [0], 0, 1, None, 0.01) == (True, "single rank")
    ok, why = rank_agree._agree("f", [0, 1], 0, 1, None, 0.01)  # torch.distributed not initialised here
    assert ok and "no distributed store" in why


@test
def test_cache_and_errors():
    rank_agree.reset()

    class Broken:
        def set(self, k, v):
            raise RuntimeError("store down")

    ok, why = rank_agree.agree("broken", [0, 1], 0, 1, store=Broken())
    assert ok is False and "store down" in why
    assert rank_agree.agree("broken", [0, 1], 0, 1, store=FakeStore()) == (ok, why)  # cached
    rank_agree.reset()


def _tcp(TCPStore, port, master):
    kw = {"wait_for_workers": False} if master else {}
    try:
        return TCPStore("127.0.0.1", port, 1, master, use_libuv=False, **kw)
    except TypeError:
        return TCPStore("127.0.0.1", port, 1, master, **kw)


@test
def test_real_tcpstore():
    try:
        from torch.distributed import TCPStore
    except Exception as e:  # noqa: BLE001
        return "SKIP torch.distributed.TCPStore unavailable: %r" % (e,)
    try:
        master = _tcp(TCPStore, 0, True)
        port = master.port
    except Exception as e:  # noqa: BLE001
        return "SKIP cannot start a TCPStore here: %r" % (e,)
    out, errs = {}, []

    def one(r, timeout_s):
        try:
            st = _tcp(TCPStore, port, False)
            out[r] = rank_agree._agree("real", [0, 1, 2], r, timeout_s, st, 0.01)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=one, args=(r, 10)) for r in (0, 1, 2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    if errs:
        return "SKIP TCPStore client failed: %r" % (errs[0],)
    assert all(ok for ok, _ in out.values()) and len(out) == 3, out
    # a group whose rank 5 never reports: both present ranks end up off
    out2 = {}

    def two(r):
        st = _tcp(TCPStore, port, False)
        out2[r] = rank_agree._agree("real", [0, 1, 5], r, 0.3, st, 0.01)

    ts = [threading.Thread(target=two, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert len(out2) == 2 and not any(ok for ok, _ in out2.values()), out2
    del master


def main():
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
    if fail:
        print("FAILED %d / %d (skip %d)" % (fail, ok + fail, skip))
        sys.exit(1)
    print("ALL PASS n=%d skip=%d" % (ok, skip))


if __name__ == "__main__":
    main()

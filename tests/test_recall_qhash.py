"""Question-hash index (qhash) of yee.core and vllm_d_recall.core.

The reply to a prompt is filed under hashes of the prompt (whole, last 512 / 128 / 32 ids). A new stream with
the same prompt - or the same prompt tail under a different head - drafts from that reply from its first token,
and lines up again inside the reply after a divergence. The disk backend keeps the index across restarts and
shares it between processes; RECALL_ACCEL_QHASH=0 / qhash=False turns it off.

Run: python tests/test_recall_qhash.py (pytest also collects the test_* functions)
"""
import importlib
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    MODS = tuple(importlib.import_module(m) for m in ("yee.core", "vllm_d_recall.core"))
except ImportError:  # bundled copy (yee/selftests): ROOT is the yee package
    sys.path.insert(0, os.path.dirname(ROOT))
    MODS = tuple(importlib.import_module(m) for m in ("yee.core", "vllm_d_recall.core"))

IDENT = {"tokenizer_digest": "a" * 64, "vocab_size": 129280, "model": "test"}


def rnd(n, seed):
    return np.random.default_rng(seed).integers(10, 120000, size=n, dtype=np.int64)


def mk(rc, directory=None, **kw):
    if directory is None:
        rec = rc.RecallRecord(None, IDENT, backend="memory", chunk_ids=1 << 14)
    else:
        rec = rc.RecallRecord(directory, IDENT, chunk_ids=1 << 14)
    return rc.RecallEngine(rec, gamma=5, async_add=False, **kw)


def serve(eng, rid, prompt, reply):
    eng.begin_stream(rid, prompt)
    eng.extend_stream(rid, reply)
    eng.end_stream(rid, reply)


def test_same_prompt_new_stream_drafts_from_first_token():
    for rc in MODS:
        eng = mk(rc)
        prompt, reply = rnd(100, 1), rnd(200, 2)
        serve(eng, "a", prompt, reply)
        eng.begin_stream("b", prompt)
        p = eng.propose("b")
        assert p is not None and p.source == "qhash", (rc.__name__, p)
        assert p.tokens.tolist() == reply[:5].tolist()
        eng.feedback("b", 5)
        eng.extend_stream("b", reply[:6])
        p = eng.propose("b")
        assert p.source == "qhash" and p.tokens.tolist() == reply[6:11].tolist()
        s = eng.stats()
        assert s["qhash"] is True and s["qhash_hits"] == 1 and s["qhash_puts"] == 1 and s["qhash_entries"] >= 1
        eng.close()


def test_different_head_same_tail_hits():
    for rc in MODS:
        eng = mk(rc)
        tail, reply = rnd(600, 3), rnd(120, 4)
        serve(eng, "a", np.concatenate((rnd(40, 5), tail)), reply)
        eng.begin_stream("b", np.concatenate((rnd(55, 6), tail)))  # another date / session id up front
        p = eng.propose("b")
        assert p is not None and p.source == "qhash" and p.tokens.tolist() == reply[:5].tolist(), rc.__name__
        eng.begin_stream("c", rnd(640, 7))  # an unrelated prompt finds nothing
        assert eng.propose("c") is None
        assert eng.stats()["qhash_hits"] == 1
        eng.close()


def test_realign_after_divergence():
    for rc in MODS:
        eng = mk(rc)
        prompt, reply = rnd(100, 8), rnd(300, 9)
        serve(eng, "a", prompt, reply)
        eng.begin_stream("b", prompt)
        eng.extend_stream("b", reply[:50])
        eng.extend_stream("b", np.array([7, 8, 9], dtype=np.int64))  # the sample left the filed reply
        st = eng._streams["b"]
        assert st.q_pos == -1
        assert eng._qhash_candidate(st, 5) is None
        eng.extend_stream("b", reply[53:56])  # 3 ids back on it: not enough yet
        assert st.q_pos == -1
        eng.extend_stream("b", reply[56:57])  # the 4th
        assert st.q_pos == 57 and eng.stats()["qhash_realigns"] == 1, (rc.__name__, st.q_pos)
        c = eng._qhash_candidate(st, 5)
        assert c is not None and c[0] == "qhash" and c[3].tolist() == reply[57:62].tolist()
        # an inserted run (the reply shifted): lines up where it left off
        eng.extend_stream("b", reply[57:80])
        eng.extend_stream("b", np.array([1, 2], dtype=np.int64))
        eng.extend_stream("b", reply[80:90])
        assert st.q_pos == 90, st.q_pos
        eng.close()


def test_end_of_reply_and_short_reply():
    for rc in MODS:
        eng = mk(rc)
        prompt, reply = rnd(100, 10), rnd(40, 11)
        serve(eng, "a", prompt, reply)
        serve(eng, "s", rnd(100, 12), rnd(4, 13))  # below QHASH_MIN_ANS: not filed
        assert eng.stats()["qhash_puts"] == 1
        eng.begin_stream("b", prompt)
        eng.extend_stream("b", reply[:37])
        c = eng._qhash_candidate(eng._streams["b"], 5)
        assert c is not None and c[4] == 3 and c[3].tolist() == reply[37:].tolist() + [int(reply[-1])] * 2
        eng.extend_stream("b", reply[37:])
        assert eng._qhash_candidate(eng._streams["b"], 5) is None
        eng.close()


def test_off_switch():
    for rc in MODS:
        prompt, reply = rnd(100, 14), rnd(200, 15)
        for eng in (mk(rc, qhash=False), mk(rc, environ={rc.ENV_NEW["qhash"]: "0"})):
            serve(eng, "a", prompt, reply)
            eng.begin_stream("b", prompt)
            p = eng.propose("b")
            assert p is None or p.source != "qhash"
            s = eng.stats()
            assert s["qhash"] is False and s["qhash_hits"] == 0 and s["qhash_entries"] == 0
            eng.close()
        eng = mk(rc, environ={rc.ENV_OLD["qhash"]: "1"})
        assert eng.stats()["qhash"] is True
        eng.close()


def test_disk_survives_restart_and_is_shared_and_cleared():
    for rc in MODS:
        d = tempfile.mkdtemp(prefix="qhash_")
        try:
            prompt, reply = rnd(100, 16), rnd(200, 17)
            eng = mk(rc, d)
            serve(eng, "a", prompt, reply)
            assert os.path.isdir(os.path.join(d, "qhash"))
            assert eng.stats()["qhash_entries"] == 2  # one file per key: the whole prompt and its last 32 ids
            other = rc._QStore(d)  # what another process sharing the directory sees
            assert other.get(rc._qhash_keys(prompt)).tolist() == reply.tolist()
            eng.close()
            eng = mk(rc, d)  # restart
            eng.begin_stream("b", prompt)
            p = eng.propose("b")
            assert p is not None and p.source == "qhash" and p.tokens.tolist() == reply[:5].tolist(), rc.__name__
            eng.end_stream("b", finished=False)
            eng.clear()
            assert eng.stats()["qhash_entries"] == 0 and other.get(rc._qhash_keys(prompt)) is None
            eng.begin_stream("c", prompt)
            assert eng._streams["c"].q_ans is None
            eng.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_store_cap_and_newest_wins():
    for rc in MODS:
        d = tempfile.mkdtemp(prefix="qhash_")
        try:
            for directory in (None, d):
                qs = rc._QStore(directory, cap=8)
                for i in range(20):
                    qs.put(["0-%024x" % i], rnd(16, 100 + i))
                    if directory:
                        os.utime(os.path.join(d, "qhash", "0-%024x.i4" % i), (1000 + i, 1000 + i))
                if directory:
                    qs._trim()
                assert qs.count() == 8, (rc.__name__, directory, qs.count())
                assert qs.get(["0-%024x" % 0]) is None
                assert qs.get(["0-%024x" % 19]).tolist() == rnd(16, 119).tolist()
                qs.put(["0-%024x" % 19], rnd(16, 999))
                assert qs.get(["0-%024x" % 19]).tolist() == rnd(16, 999).tolist()
                assert qs.put(["k"], [1, 2, 3]) == 0 and qs.put(["k"], [-1] * 16) == 0
                assert rc._QStore(directory, read_only=True).put(["k"], rnd(16, 1)) == 0
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_keys_levels():
    for rc in MODS:
        a = rnd(700, 20)
        ka = rc._qhash_keys(a)
        assert [k.split("-")[0] for k in ka] == ["0", "512", "128", "32"]
        assert [k.split("-")[0] for k in rc._qhash_keys(a[:100])] == ["0", "32"]
        kb = rc._qhash_keys(np.concatenate((rnd(9, 21), a[-600:])))
        assert ka[0] != kb[0] and ka[1:] == kb[1:]
        assert rc._qhash_keys(a.tolist()) == ka


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name, flush=True)
            except Exception:
                bad += 1
                print("FAIL %s" % name, flush=True)
                traceback.print_exc()
    print("ALL PASS" if not bad else "FAILED %d" % bad, flush=True)
    sys.exit(1 if bad else 0)

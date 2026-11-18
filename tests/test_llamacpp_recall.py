"""Tests of the llama.cpp-recall patch base (Yee 3.3.0): the MANIFEST entry, its sha256 and `yee patch list`.

Run: python tests/test_llamacpp_recall.py (also runs under pytest).
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee import patches  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


@test
def test_manifest_has_llamacpp_recall():
    m = patches.load_manifest()
    assert "llama.cpp-recall" in m["bases"], sorted(m["bases"])
    e = m["bases"]["llama.cpp-recall"]
    assert e["kind"] == "git-patch", e["kind"]
    assert e["file"] == "llama.cpp-recall/0002-recall.patch", e["file"]
    assert e["sha256"] == "203d4fe4e5d3c2251d823b3f574bf133baa9fde87423ef73cbc145587bd98024", e["sha256"]
    assert e["env"] == {"LLAMA_ARG_RECALL": "1"}, e["env"]
    assert "llama.cpp-kad" in m["bases"], sorted(m["bases"])
    assert patches.verify([{"file": e["file"], "sha256": e["sha256"]}]) == []


@test
def test_patch_list_has_llamacpp_recall():
    env = {k: v for k, v in os.environ.items()
           if not (k == "YEE" or k.startswith(("YEE_", "RECALL_ACCEL")))}
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "yee", "patch", "list"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "llama.cpp-recall" in r.stdout, r.stdout
    assert "LLAMA_ARG_RECALL=1" in r.stdout, r.stdout


def main():
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
    if fail:
        print("FAILED %d / %d" % (fail, ok + fail))
        sys.exit(1)
    print("ALL PASS n=%d" % ok)


if __name__ == "__main__":
    main()

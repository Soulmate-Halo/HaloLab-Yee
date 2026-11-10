"""Bundled tests of Yee (copied from tests/ at build time). ``yee selftest`` runs each file in a
fresh interpreter and reports PASS/FAIL per file; the files also run directly or under pytest."""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("test_recall_core.py", "test_recall_multilen.py", "test_heat.py", "test_density.py", "test_backends_equiv.py", "test_sglang_dspark.py",
         "test_sglang_heads.py", "test_vllm_hook.py", "test_vllm_heads.py", "test_tensorfold.py", "test_nongreedy.py",
         "test_yee_pkg.py")


def run_all(verbose: bool = False) -> int:
    bad = []
    env = dict(os.environ)
    for k in list(env):
        if k == "YEE" or k.startswith(("YEE_", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL", "DSV41_RECALL")):
            env.pop(k)
    for name in FILES:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            print("MISSING %s" % name, flush=True)
            bad.append(name)
            continue
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        last = [ln for ln in out.splitlines() if ln.startswith(("ALL PASS", "FAILED", "SKIP all"))]
        ok = r.returncode == 0
        print("%s %-24s %s" % ("PASS" if ok else "FAIL", name, last[-1] if last else ""), flush=True)
        if verbose or not ok:
            print(out[-6000:], flush=True)
        if not ok:
            bad.append(name)
    print("selftest %s" % ("PASS" if not bad else "FAIL: " + ", ".join(bad)), flush=True)
    return 0 if not bad else 1

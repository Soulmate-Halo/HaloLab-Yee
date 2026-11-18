"""Plain-assert tests of ``yee doctor`` (Yee 4.3): discovery is replaced by a fake ``discover.collect``; the plan,
the report, the JSON, the exit codes and the arguments handed to collect are checked. Also: a patch base with
``apply_tool: patch`` is never applied with git.

Run: python tests/test_doctor.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee import __main__ as cli  # noqa: E402
from yee import patches  # noqa: E402

try:
    from yee.integrations import discover  # noqa: E402
except Exception:  # noqa: BLE001  the module is replaced below anyway
    discover = types.ModuleType("yee.integrations.discover")
    sys.modules["yee.integrations.discover"] = discover

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def _vllm(alg="NONE"):
    return {"engine": "vllm", "role": "server", "model": "/m", "head": {"algorithm": alg},
            "where": {"host": "h", "source": "cmd"}, "argv": ["vllm", "serve", "/m"], "env": {}}


def _doctor(argv, found):
    seen = {}

    def fake_collect(**kw):
        seen.update(kw)
        return list(found)

    old = getattr(discover, "collect", None)
    discover.collect = fake_collect
    old_here = cli._doctor_engines_here
    cli._doctor_engines_here = lambda: False
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["doctor"] + list(argv))
    finally:
        cli._doctor_engines_here = old_here
        if old is not None:
            discover.collect = old
    return rc, buf.getvalue(), seen


@test
def test_no_instances_exit_3():
    rc, out, _ = _doctor(["--no-local", "--no-docker"], [])
    assert rc == 3, (rc, out)
    assert "没有找到推理服务" in out


@test
def test_vllm_without_head_exit_1():
    rc, out, _ = _doctor(["--cmd", "vllm serve /m"], [_vllm()])
    assert rc == 1, (rc, out)
    assert "部署诊断" in out and "（H0）" in out and "退出码 1" in out


@test
def test_json_output():
    rc, out, _ = _doctor(["--json"], [_vllm()])
    p = json.loads(out)
    assert p["exit"] == rc == 1
    assert p["instances"][0]["instance"]["engine"] == "vllm"
    rc, out, _ = _doctor(["--json"], [])
    assert json.loads(out)["exit"] == rc == 3


@test
def test_local_only_ignores_remote():
    _, _, seen = _doctor(["--local", "--hosts", "a,b", "--url", "http://x:1", "--k8s"], [])
    assert seen["hosts"] == [] and seen["urls"] == [] and seen["use_k8s"] is False
    assert seen["use_local"] is True


@test
def test_arguments_passed_to_collect():
    _, _, seen = _doctor(["--cmd", "a b", "--cmd", "c d", "--hosts", "h1, h2;h3", "--url", "http://u:1",
                          "--config", "m.json", "--no-docker", "--no-local", "--k8s"], [])
    assert seen["cmds"] == ["a b", "c d"], seen
    assert seen["hosts"] == ["h1", "h2", "h3"], seen
    assert seen["urls"] == ["http://u:1"] and seen["configs"] == ["m.json"]
    assert seen["use_docker"] is False and seen["use_local"] is False and seen["use_k8s"] is True


@test
def test_adapter_keeps_old_symbol_check():
    try:
        import importlib.util

        if importlib.util.find_spec("sglang") or importlib.util.find_spec("vllm"):
            return "SKIP sglang / vllm importable here"
    except Exception:  # noqa: BLE001
        pass
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["doctor", "--adapter", "vllm_spec"])
    assert rc in (1, 3), (rc, buf.getvalue())


@test
def test_apply_tool_patch_never_uses_git():
    d = tempfile.mkdtemp()
    src = os.path.join(d, "src")
    os.makedirs(src)
    pdir = os.path.join(d, "p")
    os.makedirs(os.path.join(pdir, "b"))
    body = b"--- a/x.txt\n+++ b/x.txt\n@@ -0,0 +1 @@\n+hi\n"
    with open(os.path.join(pdir, "b", "0001.patch"), "wb") as f:
        f.write(body)
    import hashlib

    entry = {"file": "b/0001.patch", "sha256": hashlib.sha256(body).hexdigest(), "kind": "git-patch",
             "apply_tool": "patch"}
    manifest = {"format": 1, "bases": {"b": entry}}
    calls = []
    old_run, old_which = patches._run, patches.shutil.which

    def fake_run(cmd, cwd):
        calls.append(list(cmd))
        return 0, ""

    patches._run = fake_run
    patches.shutil.which = lambda name: "/usr/bin/" + name
    try:
        rc, msg = patches.apply("b", src, manifest=manifest, root=pdir, check=True)
    finally:
        patches._run, patches.shutil.which = old_run, old_which
    assert calls, msg
    assert all(os.path.basename(c[0]) == "patch" for c in calls), calls
    entry.pop("apply_tool")
    calls[:] = []
    patches._run = fake_run
    patches.shutil.which = lambda name: "/usr/bin/" + name
    try:
        patches.apply("b", src, manifest=manifest, root=pdir, check=True)
    finally:
        patches._run, patches.shutil.which = old_run, old_which
    assert calls and os.path.basename(calls[0][0]) == "git", calls


@test
def test_manifest_has_maya_recall():
    e = patches.load_manifest()["bases"]["maya-recall"]
    assert e["apply_tool"] == "patch" and e["kind"] == "git-patch"
    assert e["file"] == "maya-recall/0001-maya-yee-recall.patch"
    assert e["env"] == {"RECALL_ACCEL": "1"}
    assert not patches.verify(patches.files_of(e))


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

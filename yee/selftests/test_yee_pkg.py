"""Plain-assert tests of the Yee 2.2.0 package layer: the recall_accel compatibility name, YEE_* environment
aliases, the patch MANIFEST, ``yee patch apply`` (git-patch and source-drop), ``yee doc`` and ``yee bases``.

Run: python tests/test_yee_pkg.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

import yee  # noqa: E402
from yee import __main__ as cli  # noqa: E402
from yee import patches  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def _py(code: str, extra_env=None):
    env = {k: v for k, v in os.environ.items()
           if not (k == "YEE" or k.startswith(("YEE_", "RECALL_ACCEL")))}
    if os.path.isdir(os.path.join(ROOT, "yee")):
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


@test
def test_compat_same_modules():
    out = _py("import sys, recall_accel, yee\n"
              "import recall_accel.integrations.sglang_dspark as a\n"
              "import yee.integrations.sglang_dspark as b\n"
              "assert a is b\n"
              "assert sys.modules['recall_accel.integrations.sglang_dspark'] is "
              "sys.modules['yee.integrations.sglang_dspark']\n"
              "assert recall_accel is yee and recall_accel.__version__ == '2.2.0'\n"
              "from recall_accel.core import RecallEngine\n"
              "import yee.core\n"
              "assert RecallEngine is yee.core.RecallEngine\n"
              "assert b.__spec__.name == 'yee.integrations.sglang_dspark'\n"
              "print('ok')")
    assert out.endswith("ok"), out


@test
def test_compat_autoload_old_name():
    # the 2.x .pth line; with RECALL_ACCEL unset it must import quietly and install nothing
    out = _py("import sys, recall_accel.autoload as a, yee.autoload as b\n"
              "assert a is b and not a.enabled()\n"
              "print('ok')")
    assert out.endswith("ok"), out


@test
def test_yee_env_alias():
    out = _py("import os, yee\n"
              "print(os.environ.get('RECALL_ACCEL_GIB'), os.environ.get('RECALL_ACCEL'), "
              "os.environ.get('RECALL_ACCEL_DIR'))",
              {"YEE_GIB": "0.5", "YEE": "1", "YEE_DIR": "/new", "RECALL_ACCEL_DIR": "/old"})
    assert out == "0.5 1 /old", out  # YEE_* fills in, an explicit RECALL_ACCEL_* still wins
    env = {"YEE_MIN_MATCH": "12"}
    yee._alias_env(env)
    assert env["RECALL_ACCEL_MIN_MATCH"] == "12"


@test
def test_manifest_sha256():
    m = patches.load_manifest()
    assert set(m["bases"]) == {"tensorfold", "cuteafd", "mlx-serve"}, sorted(m["bases"])
    seen = set()
    for name, e in m["bases"].items():
        files = []
        if e.get("variants"):
            files += list(e["variants"].values())
        elif e["kind"] == "git-patch":
            files.append({"file": e["file"], "sha256": e["sha256"]})
        files += list(e.get("files") or [])
        if e.get("notes"):
            files.append(e["notes"])
        assert files, name
        assert patches.verify(files) == [], (name, patches.verify(files))
        seen.update(f["file"] for f in files)
    shipped = set()
    for dp, _, fns in os.walk(patches.PATCH_DIR):
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), patches.PATCH_DIR).replace(os.sep, "/")
            if rel != "MANIFEST.json":
                shipped.add(rel)
    assert shipped == seen, (sorted(shipped - seen), sorted(seen - shipped))
    assert m["bases"]["tensorfold"]["env"] == {"TF_RECALL_AFTER_PRICED": "1"}
    assert m["bases"]["cuteafd"]["default_variant"] == "v2"
    assert len(m["bases"]["mlx-serve"]["files"]) == 4


@test
def test_git_patch_apply():
    git = shutil.which("git")
    if not git:
        return "SKIP no git on PATH"
    with tempfile.TemporaryDirectory() as td:
        repo = os.path.join(td, "repo")
        os.makedirs(repo)
        subprocess.run([git, "init", "-q"], cwd=repo, check=True)
        with open(os.path.join(repo, "a.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("hello\nworld\n")
        pf = os.path.join(td, "t.patch")
        with open(pf, "w", encoding="utf-8", newline="\n") as f:
            f.write("diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-hello\n+hello yee\n"
                    " world\n")
        m = {"bases": {"t": {"kind": "git-patch", "file": pf, "sha256": patches.sha256_file(pf)}}}
        rc, msg = patches.apply("t", repo, check=True, manifest=m)
        assert rc == 0 and msg.startswith("check ok"), (rc, msg)
        assert open(os.path.join(repo, "a.txt"), encoding="utf-8").read() == "hello\nworld\n"
        rc, msg = patches.apply("t", repo, manifest=m)
        assert rc == 0 and msg.startswith("applied"), (rc, msg)
        assert open(os.path.join(repo, "a.txt"), encoding="utf-8").read() == "hello yee\nworld\n"
        rc, msg = patches.apply("t", repo, manifest=m)
        assert rc == 0 and msg.startswith("already applied"), (rc, msg)
        # a tree the patch does not fit: check fails with 1
        with open(os.path.join(repo, "a.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("other\n")
        rc, msg = patches.apply("t", repo, manifest=m)
        assert rc == 1, (rc, msg)
        # a patch file that does not match its sha256 is refused
        bad = {"bases": {"t": {"kind": "git-patch", "file": pf, "sha256": "0" * 64}}}
        rc, msg = patches.apply("t", repo, manifest=bad)
        assert rc == 1 and "refusing" in msg, (rc, msg)


@test
def test_source_drop():
    with tempfile.TemporaryDirectory() as td:
        new = os.path.join(td, "x.zig")
        with open(new, "w", encoding="utf-8", newline="\n") as f:
            f.write("const new = 1;\n")
        src = os.path.join(td, "src_root")
        spec = os.path.join(src, "src", "engines", "mlx", "spec")
        os.makedirs(spec)
        with open(os.path.join(spec, "x.zig"), "w", encoding="utf-8", newline="\n") as f:
            f.write("const old = 1;\n")
        m = {"bases": {"d": {"kind": "source-drop", "target": "src/engines/mlx/spec",
                             "files": [{"file": new, "sha256": patches.sha256_file(new),
                                        "dest": "src/engines/mlx/spec/x.zig"}]}}}
        rc, msg = patches.apply("d", src, check=True, manifest=m)
        assert rc == 0 and msg.startswith("check ok"), (rc, msg)
        assert open(os.path.join(spec, "x.zig"), encoding="utf-8").read() == "const old = 1;\n"
        rc, msg = patches.apply("d", src, manifest=m)
        assert rc == 0 and msg.startswith("replaced 1"), (rc, msg)
        assert open(os.path.join(spec, "x.zig"), encoding="utf-8").read() == "const new = 1;\n"
        backups = [d for d in os.listdir(src) if d.startswith(".yee-backup-")]
        assert len(backups) == 1, backups
        kept = os.path.join(src, backups[0], "src", "engines", "mlx", "spec", "x.zig")
        assert open(kept, encoding="utf-8").read() == "const old = 1;\n"
        rc, msg = patches.apply("d", src, manifest=m)
        assert rc == 0 and msg.startswith("already applied"), (rc, msg)
        empty = os.path.join(td, "empty")
        os.makedirs(empty)
        rc, msg = patches.apply("d", empty, manifest=m)
        assert rc == 1, (rc, msg)
        try:
            patches.apply("nope", src, manifest=m)
            raise AssertionError("unknown base accepted")
        except patches.PatchUsageError:
            pass


@test
def test_patch_cli_exit_codes():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["patch", "list"]) == 0
        assert cli.main(["patch", "path", "tensorfold"]) == 0
        assert cli.main(["patch", "show", "mlx-serve"]) == 0
        assert cli.main(["patch", "apply", "nope", "--src", "."]) == 2
    out = buf.getvalue()
    for word in ("tensorfold", "cuteafd", "mlx-serve", "TF_RECALL_AFTER_PRICED=1", "0001-recall-after-priced.patch",
                 "d_recall.zig"):
        assert word in out, (word, out)


@test
def test_doc_and_bases():
    topics = cli.doc_topics()
    assert "index" in topics, topics
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["doc"]) == 0
        assert cli.main(["doc", "index"]) == 0
        assert cli.main(["doc", "no-such-topic"]) == 2
        assert cli.main(["bases"]) == 0
        assert cli.main(["bases", "--json"]) == 0
    out = buf.getvalue()
    for word in ("index", "tensorfold", "cuteafd", "mlx-serve", "DSPARK"):
        assert word in out, (word, out[:2000])
    rows = cli.bases_rows()
    assert {r["base"] for r in rows} >= {"sglang", "vllm", "tensorfold", "cuteafd", "mlx-serve"}
    json.dumps(rows)


@test
def test_tensorfold_launch_env():
    from yee.integrations import tensorfold

    assert tensorfold.launch_env() == {"TF_RECALL_AFTER_PRICED": "1"}
    err = io.StringIO()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert tensorfold.main(["args", "--no-detect"]) == 0
    assert out.getvalue().strip().startswith("--recall"), out.getvalue()
    assert "export TF_RECALL_AFTER_PRICED=1" in err.getvalue(), err.getvalue()


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

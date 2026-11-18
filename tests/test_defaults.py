"""Plain-assert tests of the Yee 4.2 startup defaults (``yee.integrations.defaults``), the sitecustomize template
(defaults before Mia's sitecustomize, fallback without the yee package) and ``yee config``.

Run: python tests/test_defaults.py (also runs under pytest and in ``yee selftest``).
"""
from __future__ import annotations

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
from yee.integrations import defaults  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(defaults.__file__)), "sitecustomize_template.py")
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(yee.__file__)))  # the directory holding the yee package
RESULTS = []
ON = {"RECALL_ACCEL": "1", "YEE_KAD": "1", "YEE_EXPERT_BUDGET": "1", "YEE_EXPERT_BUDGET_ARMED": "1",
      "YEE_EXPERT_BUDGET_P": "0.9", "VLLM_D_RECALL": "1", "VLLM_D_RECALL_NONGREEDY": "1", "VLLM_D_RECALL_KAD": "1",
      "RECALL_ACCEL_NONGREEDY": "1", "RECALL_ACCEL_GIB": "0.25", "RECALL_ACCEL_INDEX_DENSITY": "1"}


def test(fn):
    RESULTS.append(fn)
    return fn


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not (k == "YEE" or k.startswith(("YEE_", "RECALL_ACCEL", "SGLANG_D_RECALL", "VLLM_D_RECALL",
                                               "DSV41_", "DSPARK_", "SGLANG_DSPARK")))}
    env.pop("PYTHONPATH", None)
    env.update(extra)
    return env


@test
def test_defaults_fill_everything_when_unset():
    env = {}
    out = defaults.apply(env, argv=[])
    for k, v in ON.items():
        assert env.get(k) == v, (k, env.get(k))
    assert out == env or set(out) <= set(env)
    assert defaults.apply(env, argv=[]) == {}  # idempotent


@test
def test_master_off_sets_nothing():
    for name in defaults.MASTER:
        for off in ("0", "off", "false", "no", "OFF"):
            env = {name: off}
            assert defaults.apply(env, argv=["x", "--speculative-dspark-block-size", "7"]) == {}
            assert env == {name: off}


@test
def test_explicit_values_and_aliases_win():
    env = {"YEE": "1", "YEE_EXPERT_BUDGET": "0", "SGLANG_D_RECALL_GIB": "0.5", "YEE_INDEX_DENSITY": "0.5",
           "YEE_NONGREEDY": "0", "YEE_EXPERT_BUDGET_P": "0.95"}
    defaults.apply(env, argv=[])
    assert "RECALL_ACCEL" not in env  # YEE=1 already counts
    assert env["YEE_EXPERT_BUDGET"] == "0" and env["YEE_EXPERT_BUDGET_P"] == "0.95"
    assert "RECALL_ACCEL_GIB" not in env and "VLLM_D_RECALL_GIB" not in env
    assert "RECALL_ACCEL_INDEX_DENSITY" not in env and "RECALL_ACCEL_NONGREEDY" not in env
    assert env["YEE_KAD"] == "1" and env["VLLM_D_RECALL"] == "1"
    env = {"RECALL_ACCEL": ""}  # empty counts as unset
    defaults.apply(env, argv=[])
    assert env["RECALL_ACCEL"] == "1"


@test
def test_mia_fills_verify_cap_from_launch_args():
    with tempfile.TemporaryDirectory() as d:
        site = os.path.join(d, "sitecustomize.py")
        open(site, "w").close()
        for argv in (["x", "--speculative-dspark-block-size", "7"], ["x", "--speculative-dspark-block-size=7"]):
            env = {"SGLANG_D_RECALL_MIA_SITECUSTOMIZE": site}
            out = defaults.apply(env, argv=argv)
            assert env["DSPARK_BLOCK_SIZE"] == "7" and env["DSV41_VERIFY_CAP"] == "conf:0.1", env
            assert env["DSV41_VERIFY_CAP_STRIDE"] == "8" and env["SGLANG_DSPARK_FOLDED_SAMPLING"] == "2"
            assert out["DSPARK_BLOCK_SIZE"] == "7"
        env = {"SGLANG_D_RECALL_MIA_SITECUSTOMIZE": site}
        defaults.apply(env, argv=["x"])  # block size unknown: no cap, the stride would be wrong
        assert "DSV41_VERIFY_CAP" not in env and "DSPARK_BLOCK_SIZE" not in env
        assert env["SGLANG_DSPARK_FOLDED_SAMPLING"] == "2"
        env = {"SGLANG_D_RECALL_MIA_SITECUSTOMIZE": site, "DSPARK_BLOCK_SIZE": "12", "DSV41_VERIFY_CAP": "3",
               "SGLANG_DSPARK_FOLDED_SAMPLING": "0"}
        defaults.apply(env, argv=["x", "--speculative-dspark-block-size", "7"])
        assert env["DSPARK_BLOCK_SIZE"] == "12" and env["DSV41_VERIFY_CAP"] == "3"
        assert env["DSV41_VERIFY_CAP_STRIDE"] == "13" and env["SGLANG_DSPARK_FOLDED_SAMPLING"] == "0"
    assert defaults.block_size_from_argv(["--speculative-dspark-block-size", "x"]) is None
    assert defaults.block_size_from_argv(["--speculative-dspark-block-size", "0"]) is None
    assert defaults.block_size_from_argv([]) is None


@test
def test_non_mia_leaves_engine_variables_alone():
    if os.path.isdir(defaults.MIA_ADAPTER):
        return "SKIP inside a Mia image"
    env = {}
    defaults.apply(env, argv=["x", "--speculative-dspark-block-size", "7"])
    assert not any(k.startswith(("DSV41_", "DSPARK_", "SGLANG_DSPARK")) for k in env), env


@test
def test_effective_sources():
    rows = {n: (v, s) for n, v, s in defaults.effective({"YEE_KAD": "0", "SGLANG_D_RECALL_GIB": "0.5"})}
    assert rows["YEE_KAD"] == ("0", "env")
    assert rows["YEE_EXPERT_BUDGET"] == ("1", "default") and rows["YEE / RECALL_ACCEL"] == ("1", "default")
    assert rows["RECALL_ACCEL_GIB (SGLANG_D_RECALL_GIB)"] == ("0.5", "env")
    rows = {n: (v, s) for n, v, s in defaults.effective({"YEE": "0"})}
    assert rows["YEE / RECALL_ACCEL"] == ("0", "env") and rows["YEE_KAD"] == ("0", "off")
    env = {"YEE_KAD": "0"}
    defaults.apply(env, argv=[])  # what apply() filled stays "default" (e.g. sitecustomize ran in this process)
    rows = {n: (v, s) for n, v, s in defaults.effective(env)}
    assert rows["YEE / RECALL_ACCEL"] == ("1", "default") and rows["YEE_EXPERT_BUDGET_P"] == ("0.9", "default")
    assert rows["YEE_KAD"] == ("0", "env")


@test
def test_cli_config_json():
    env = _clean_env(YEE_KAD="0", PYTHONPATH=PKG_ROOT)
    r = subprocess.run([sys.executable, "-m", "yee", "config", "--json"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = {x["name"]: x for x in json.loads(r.stdout)}
    assert rows["YEE_KAD"]["source"] == "env" and rows["YEE_KAD"]["value"] == "0"
    assert rows["YEE_EXPERT_BUDGET"]["source"] == "default"
    r = subprocess.run([sys.executable, "-m", "yee", "config"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "YEE_EXPERT_BUDGET_P" in r.stdout and yee.__version__ in r.stdout


def _run_template(pythonpath, mia, mark, fake_yee=False):
    with tempfile.TemporaryDirectory() as d:
        site_dir = os.path.join(d, "site")
        os.makedirs(site_dir)
        shutil.copyfile(TEMPLATE, os.path.join(site_dir, "sitecustomize.py"))
        if fake_yee:  # shadow any installed yee so the template's own fallback has to work
            os.makedirs(os.path.join(site_dir, "yee"))
            with open(os.path.join(site_dir, "yee", "__init__.py"), "w") as f:
                f.write("raise ImportError('yee is broken here on purpose')\n")
        env = _clean_env(PYTHONPATH=os.pathsep.join([site_dir] + pythonpath), SGLANG_D_RECALL_MIA_SITECUSTOMIZE=mia,
                         MARK=mark, RECALL_ACCEL_ADAPTER="off", YEE_KAD="0")
        code = "import os; print('EB=%s P=%s' % (os.environ.get('YEE_EXPERT_BUDGET'), os.environ.get('YEE_EXPERT_BUDGET_P')))"
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


@test
def test_template_defaults_run_before_mia():
    with tempfile.TemporaryDirectory() as d:
        mia = os.path.join(d, "mia_site.py")
        mark = os.path.join(d, "mark.txt")
        with open(mia, "w") as f:
            f.write("import os\nopen(os.environ['MARK'], 'w').write(os.environ.get('YEE_EXPERT_BUDGET', '-') + '|' + "
                    "os.environ.get('RECALL_ACCEL', '-'))\n")
        r = _run_template([PKG_ROOT], mia, mark)
        assert r.returncode == 0, r.stderr[-2000:]
        assert open(mark).read() == "1|1", open(mark).read()
        assert "EB=1 P=0.9" in r.stdout and "[yee] 4.2.0 defaults:" in r.stdout, r.stdout
        os.remove(mark)
        r = _run_template([], mia, mark, fake_yee=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert open(mark).read() == "1|1" and "EB=1 P=0.9" in r.stdout, r.stdout


@test
def test_template_respects_yee_off():
    with tempfile.TemporaryDirectory() as d:
        mia = os.path.join(d, "mia_site.py")
        mark = os.path.join(d, "mark.txt")
        with open(mia, "w") as f:
            f.write("import os\nopen(os.environ['MARK'], 'w').write(os.environ.get('YEE_EXPERT_BUDGET', '-'))\n")
        with tempfile.TemporaryDirectory() as d2:
            site_dir = os.path.join(d2, "site")
            os.makedirs(site_dir)
            shutil.copyfile(TEMPLATE, os.path.join(site_dir, "sitecustomize.py"))
            env = _clean_env(PYTHONPATH=os.pathsep.join([site_dir, PKG_ROOT]), SGLANG_D_RECALL_MIA_SITECUSTOMIZE=mia,
                             MARK=mark, YEE="0")
            r = subprocess.run([sys.executable, "-c", "print('ok')"], env=env, capture_output=True, text=True,
                               timeout=120)
        assert r.returncode == 0 and open(mark).read() == "-", (r.stdout, r.stderr)
        assert "[yee-eb]" not in r.stdout and "[recall" not in r.stdout


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

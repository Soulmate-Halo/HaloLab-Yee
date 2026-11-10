"""Command line of sglang-d-recall: ``python -m sglang_d_recall <subcommand>``.

Subcommands:
  install-hook     write a sglang_d_recall.pth into the current interpreter's site-packages
  uninstall-hook   remove that .pth again
  doctor           check the interpreter, numpy / torch / sglang and the SGLang symbols the hooks need
  status           print the stats.json of the record directory
  pause | on|off   create / remove the "paused" file in the record directory
  clear            write the "clear.request" file in the record directory
  selftest         run a few built-in cases (no pytest needed)

The record directory is resolved exactly like ``core.engine_from_env`` (SGLANG_D_RECALL_DIR, else
~/.cache/sglang_d_recall/<tokenizer digest, first 16 chars>); ``--dir`` overrides it.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import sys
import sysconfig
import tempfile

from . import __version__
from .core import ENV_NEW, ENV_OLD, _env

PTH_NAME = "sglang_d_recall.pth"
PTH_LINE = "import sglang_d_recall.autoload"

# The SGLang symbols sglang_dspark.py patches: module path -> (class, attributes/methods).
REQUIRED_SYMBOLS = (
    ("sglang.srt.speculative.dspark_components.dspark_worker_v2", "DSparkWorkerV2",
     ("__init__", "_proposer", "propose")),
    ("sglang.srt.managers.scheduler_components.batch_result_processor", "SchedulerBatchResultProcessor",
     ("process_batch_result_decode",)),
    ("sglang.srt.models.deepseek_v4_dspark", "DeepseekV4ForCausalLMDSpark",
     ("compute_confidence", "gamma")),
    ("sglang.srt.distributed", None, ("get_tp_group",)),
)


# ----------------------------------------------------------------------------------------------- helpers
def _site_packages() -> str:
    """The site-packages directory of the running interpreter.

    The directory the package itself lives in wins (a --user install must get its .pth next to it),
    otherwise the interpreter's purelib is used.
    """
    candidates = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(purelib)
    try:
        candidates.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001  not available in every environment
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    seen = set()
    uniq = []
    for p in candidates:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    for p in uniq:
        if os.path.isdir(os.path.join(p, "sglang_d_recall")):
            return p
    for p in uniq:
        if p and os.path.isdir(p):
            return p
    return purelib or os.path.join(sys.prefix, "Lib", "site-packages")


def _pth_path() -> str:
    return os.path.join(_site_packages(), PTH_NAME)


def _tokenizer_digest() -> str:
    """The digest of the configured tokenizer directory; the placeholder digest when there is none."""
    d = _env(ENV_NEW["tokenizer_dir"], ENV_OLD["tokenizer_dir"], "/models/DeepSeek-V4.1-Flash")
    try:
        from .integrations.sglang_dspark import _digest_from_dir

        digest, _, found = _digest_from_dir(d)
        if found:
            return digest
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _record_dir(override: str = "") -> str:
    """The record directory: --dir, then SGLANG_D_RECALL_DIR, then ~/.cache/sglang_d_recall/<digest[:16]>."""
    if override:
        return os.path.abspath(os.path.expanduser(override))
    d = _env(ENV_NEW["dir"], ENV_OLD["dir"], "")
    if d:
        return os.path.abspath(os.path.expanduser(d))
    return os.path.join(os.path.expanduser("~/.cache/sglang_d_recall"), _tokenizer_digest()[:16])


def _env_report() -> list:
    """The environment as it is in effect right now: new name first, old name as fallback."""
    rows = []
    for key in ("on", "dir", "gib", "backend", "device", "min_match", "prior", "read_only", "min_free_gib",
                "tokenizer_dir", "model_name", "mia_sitecustomize", "evict", "hot_frac", "dedup_run",
                "heat_halflife", "heat_seen_weight", "promote_tokens", "index_density"):
        new, old = ENV_NEW[key], ENV_OLD[key]
        rows.append((key, new, old, _env(new, old, "")))
    return rows


# ----------------------------------------------------------------------------------------------- commands
def cmd_install_hook(args) -> int:
    p = _pth_path()
    existed = os.path.exists(p)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(PTH_LINE + "\n")
    except OSError as e:
        print("install-hook FAILED: cannot write %s: %s" % (p, e), flush=True)
        return 1
    print("%s %s" % ("overwrote" if existed else "wrote", p), flush=True)
    print("content: %s" % (PTH_LINE,), flush=True)
    return 0


def cmd_uninstall_hook(args) -> int:
    p = _pth_path()
    if not os.path.exists(p):
        print("no hook at %s" % (p,), flush=True)
        return 0
    try:
        os.remove(p)
    except OSError as e:
        print("uninstall-hook FAILED: cannot remove %s: %s" % (p, e), flush=True)
        return 1
    print("removed %s" % (p,), flush=True)
    return 0


def cmd_doctor(args) -> int:
    missing = []
    print("sglang-d-recall %s" % (__version__,), flush=True)
    print("python %s (%s)" % (sys.version.split()[0], sys.executable), flush=True)

    try:
        import numpy

        print("numpy OK %s" % (numpy.__version__,), flush=True)
    except Exception as e:  # noqa: BLE001
        print("numpy MISSING (%r)" % (e,), flush=True)
        missing.append("numpy")

    try:
        import torch

        print("torch OK %s" % (torch.__version__,), flush=True)
    except Exception as e:  # noqa: BLE001
        print("torch MISSING (%r)" % (e,), flush=True)

    try:
        import sglang

        ver = getattr(sglang, "__version__", None)
        print("sglang OK %s" % (ver or "<no __version__>",), flush=True)
    except Exception as e:  # noqa: BLE001
        print("sglang NOT IMPORTABLE (%r)" % (e,), flush=True)
        missing.append("sglang")

    print("SGLang symbols sglang_dspark.py needs:", flush=True)
    for mod_path, cls_name, attrs in REQUIRED_SYMBOLS:
        try:
            mod = __import__(mod_path, fromlist=["__name__"])
        except Exception as e:  # noqa: BLE001
            print("  MISSING module %s (%r)" % (mod_path, e), flush=True)
            missing.append("%s (module)" % (mod_path,))
            continue
        if cls_name is not None:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                print("  MISSING %s.%s" % (mod_path, cls_name), flush=True)
                missing.append("%s.%s" % (mod_path, cls_name))
                continue
            bad = [a for a in attrs if not hasattr(cls, a)]
            if bad:
                print("  MISSING %s.%s: %s" % (mod_path, cls_name, ", ".join(bad)), flush=True)
                missing.extend("%s.%s.%s" % (mod_path, cls_name, a) for a in bad)
            else:
                print("  OK %s.%s (%s)" % (mod_path, cls_name, ", ".join(attrs)), flush=True)
        else:
            bad = [a for a in attrs if not hasattr(mod, a)]
            if bad:
                print("  MISSING %s: %s" % (mod_path, ", ".join(bad)), flush=True)
                missing.extend("%s.%s" % (mod_path, a) for a in bad)
            else:
                print("  OK %s (%s)" % (mod_path, ", ".join(attrs)), flush=True)

    p = _pth_path()
    hook = os.path.exists(p)
    print("hook: %s (%s)" % ("installed" if hook else "not installed", p), flush=True)

    print("environment in effect:", flush=True)
    for key, new, old, value in _env_report():
        print("  %-16s %-32s %-28s = %s" % (key, new, old, value if value else "<unset>"), flush=True)

    if missing:
        print("missing: %s" % (", ".join(missing),), flush=True)
        print("doctor FAIL", flush=True)
        return 1
    print("doctor PASS", flush=True)
    return 0


def cmd_status(args) -> int:
    d = _record_dir(args.dir)
    p = os.path.join(d, "stats.json")
    print("record dir: %s" % (d,), flush=True)
    if not os.path.exists(p):
        print("no stats.json in %s" % (d,), flush=True)
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print("cannot read %s: %r" % (p, e), flush=True)
        return 1
    for k in sorted(data):
        print("  %-18s %s" % (k, data[k]), flush=True)
    return 0


def cmd_pause(args) -> int:
    d = _record_dir(args.dir)
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "paused")
    if args.state == "on":
        with open(p, "w", encoding="utf-8") as f:
            f.write("")
        print("paused: wrote %s" % (p,), flush=True)
    else:
        if os.path.exists(p):
            os.remove(p)
        print("paused: removed %s" % (p,), flush=True)
    return 0


def cmd_clear(args) -> int:
    d = _record_dir(args.dir)
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "clear.request")
    with open(p, "w", encoding="utf-8") as f:
        f.write("")
    print("clear requested: wrote %s" % (p,), flush=True)
    return 0


def cmd_selftest(args) -> int:
    """A few built-in cases: memory / disk backends (add, propose, evict) and the env new/old fallback."""
    import numpy as np

    from . import core as rc

    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:  # noqa: BLE001
            checks.append((name, False, "%s: %s" % (type(e).__name__, e)))

    ident = {"tokenizer_digest": "a" * 64, "vocab_size": 128000, "model": "selftest"}

    def backend_case(backend):
        def run():
            tmp = tempfile.mkdtemp(prefix="sglang_d_recall_st_")
            try:
                d = os.path.join(tmp, "rec")
                # enough capacity: the ids stay in the record
                r = rc.RecallRecord(d if backend == "disk" else None, ident, gib=1024 * 64 / float(1 << 30),
                                    chunk_ids=256, backend=backend)
                seq = np.random.default_rng(7).integers(10, 90000, size=1200, dtype=np.int64)
                added = r.add(seq)
                assert added > 0, "add returned %r" % (added,)
                hit = r.lookup(seq[100:140])
                assert hit is not None, "lookup found nothing"
                p, L = hit
                assert L >= rc.WINDOW, L
                cont = r.continuation(p, 5, min_n=1)
                assert cont is not None and cont.size > 0, "no continuation"
                st = r.stats()
                assert st["ids"] > 0 and st["backend"] == backend, st
                r.close()
                # evict: over the capacity the oldest chunk goes first
                r2 = rc.RecallRecord(d if backend == "disk" else None, ident, gib=1024 * 4 / float(1 << 30),
                                     chunk_ids=256, backend=backend)
                for k in range(6):
                    r2.add(np.random.default_rng(8 + k).integers(10, 90000, size=400, dtype=np.int64))
                assert r2.evicted_chunks > 0, "nothing was evicted"
                r2.clear()
                assert r2.end_pos - r2.base_pos == 0, "clear did not empty the record"
                r2.close()
            finally:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)
        return run

    def engine_case():
        tmp = tempfile.mkdtemp(prefix="sglang_d_recall_st_")
        try:
            d = os.path.join(tmp, "rec")
            r = rc.RecallRecord(d, ident, chunk_ids=1 << 13)
            eng = rc.RecallEngine(r, gamma=5, async_add=False)
            prompt = np.random.default_rng(3).integers(10, 90000, size=200, dtype=np.int64)
            reply = np.random.default_rng(4).integers(10, 90000, size=300, dtype=np.int64)
            eng.begin_stream("a", prompt)
            eng.end_stream("a", reply)
            eng.begin_stream("b", prompt)
            eng.extend_stream("b", reply[:20])
            prop = eng.propose("b")
            assert prop is not None, "no proposal for a repeated request"
            assert prop.tokens.size == 5, prop.tokens.size
            eng.feedback("b", 3)
            eng.close()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def env_case():
        # new name wins
        env = {ENV_NEW["on"]: "1", ENV_NEW["dir"]: "/tmp/x", ENV_NEW["backend"]: "memory",
               ENV_OLD["on"]: "0", ENV_OLD["backend"]: "disk"}
        assert rc._env(ENV_NEW["on"], ENV_OLD["on"], "0", env) == "1"
        assert rc._env(ENV_NEW["backend"], ENV_OLD["backend"], "disk", env) == "memory"
        # old name is the fallback
        env = {ENV_OLD["on"]: "1", ENV_OLD["dir"]: "/tmp/y", ENV_OLD["backend"]: "memory"}
        assert rc._env(ENV_NEW["on"], ENV_OLD["on"], "0", env) == "1"
        assert rc._env(ENV_NEW["backend"], ENV_OLD["backend"], "disk", env) == "memory"
        # default when nothing is set
        assert rc._env(ENV_NEW["gib"], ENV_OLD["gib"], "0.0625", {}) == "0.0625"

    def env_engine_case():
        rc._ENGINE = None
        tmp = tempfile.mkdtemp(prefix="sglang_d_recall_st_")
        try:
            env = {ENV_OLD["on"]: "1", ENV_OLD["dir"]: tmp, ENV_OLD["gib"]: "0.01",
                   ENV_OLD["backend"]: "memory"}
            eng = rc.engine_from_env(5, ident, environ=env)
            assert eng is not None, "the old names did not switch the engine on"
            assert eng.record.backend == "memory", eng.record.backend
            eng.close()
            rc._ENGINE = None
            assert rc.engine_from_env(5, ident, environ={ENV_NEW["on"]: "0"}) is None
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    check("memory backend: add/lookup/continuation/evict/clear", backend_case("memory"))
    check("disk backend: add/lookup/continuation/evict/clear", backend_case("disk"))
    check("engine: propose from the record", engine_case)
    check("env: new name wins, old name is the fallback", env_case)
    check("env: engine_from_env reads the old names", env_engine_case)

    ok = True
    for name, passed, err in checks:
        if passed:
            print("PASS %s" % (name,), flush=True)
        else:
            ok = False
            print("FAIL %s: %s" % (name, err), flush=True)
    if ok:
        print("selftest PASS", flush=True)
        return 0
    print("selftest FAIL", flush=True)
    return 1


# ----------------------------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sglang-d-recall", description="SGLang decode(D) side Recall accelerator")
    p.add_argument("--version", action="version", version="sglang-d-recall %s" % (__version__,))
    sub = p.add_subparsers(dest="cmd")

    def add_dir(sp):
        sp.add_argument("--dir", default="", help="record directory (default: like core.engine_from_env)")

    sp = sub.add_parser("install-hook", help="write the .pth into this interpreter's site-packages")
    sp.set_defaults(func=cmd_install_hook)

    sp = sub.add_parser("uninstall-hook", help="remove that .pth again")
    sp.set_defaults(func=cmd_uninstall_hook)

    sp = sub.add_parser("doctor", help="check the interpreter and the SGLang symbols the hooks need")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("status", help="print the stats.json of the record directory")
    add_dir(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("pause", help="pause / resume the copies (the 'paused' file)")
    add_dir(sp)
    sp.add_argument("state", choices=("on", "off"), help="on = pause, off = resume")
    sp.set_defaults(func=cmd_pause)

    sp = sub.add_parser("clear", help="ask the record to be cleared (the 'clear.request' file)")
    add_dir(sp)
    sp.set_defaults(func=cmd_clear)

    sp = sub.add_parser("selftest", help="run the built-in cases (no pytest needed)")
    sp.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

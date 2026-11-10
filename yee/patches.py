"""Source patches shipped inside Yee for the bases that are not Python hooks: TensorFold, cuteafd, mlx-serve.

``patches/MANIFEST.json`` lists every base: kind (``git-patch`` or ``source-drop``), baseline, the files with
their sha256 (written at build time), variants and launch env. ``yee patch list | show | path | apply``.

apply, git-patch:   sha256 check -> ``git apply --reverse --check`` (already applied: say so, rc 0) ->
                    ``git apply --check`` -> ``git apply``; without git the same with ``patch -p1``.
apply, source-drop: sha256 check -> DIR/<target> must exist -> files that differ are backed up to
                    DIR/.yee-backup-<timestamp>/ and then overwritten.
``--check`` only checks and writes nothing. Exit codes: 0 ok, 1 check failed, 2 usage error.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(HERE, "patches")
MANIFEST = os.path.join(PATCH_DIR, "MANIFEST.json")
RC_OK, RC_CHECK, RC_USAGE = 0, 1, 2


class PatchUsageError(Exception):
    """Unknown base / variant, missing source directory: exit code 2."""


def load_manifest(path: str = "") -> dict:
    with open(path or MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(rel: str, root: str = "") -> str:
    """A manifest file path: absolute paths are used as they are, relative ones live under the patch dir."""
    return rel if os.path.isabs(rel) else os.path.join(root or PATCH_DIR, *rel.split("/"))


def _base(manifest: dict, base: str) -> dict:
    bases = manifest.get("bases") or {}
    if base not in bases:
        raise PatchUsageError("unknown base %r (have: %s)" % (base, ", ".join(sorted(bases)) or "none"))
    return bases[base]


def variant_entry(entry: dict, variant=None) -> dict:
    """The {file, sha256} of a git-patch entry for the chosen variant (default_variant when None)."""
    variants = entry.get("variants") or {}
    if not variants:
        if variant:
            raise PatchUsageError("this base has no variants (got --variant %s)" % variant)
        return {"file": entry["file"], "sha256": entry["sha256"]}
    v = variant or entry.get("default_variant") or sorted(variants)[0]
    if v not in variants:
        raise PatchUsageError("unknown variant %r (have: %s)" % (v, ", ".join(sorted(variants))))
    return variants[v]


def files_of(entry: dict, variant=None) -> list:
    """Every {file, sha256[, dest]} the chosen base/variant needs."""
    if entry.get("kind") == "source-drop":
        if variant:
            raise PatchUsageError("source-drop bases have no variants (got --variant %s)" % variant)
        return list(entry.get("files") or [])
    return [variant_entry(entry, variant)]


def verify(files: list, root: str = "") -> list:
    """Problems as text (empty list = every file exists and matches its sha256)."""
    bad = []
    for it in files:
        p = resolve(it["file"], root)
        if not os.path.isfile(p):
            bad.append("missing %s" % p)
        elif sha256_file(p) != str(it.get("sha256", "")).lower():
            bad.append("sha256 mismatch %s" % p)
    return bad


def list_rows(manifest=None) -> list:
    m = manifest or load_manifest()
    rows = []
    for name in sorted(m.get("bases") or {}):
        e = m["bases"][name]
        rows.append({"base": name, "kind": e.get("kind"), "variants": sorted(e.get("variants") or {}),
                     "default_variant": e.get("default_variant"), "baseline": e.get("baseline"),
                     "env": e.get("env") or {}})
    return rows


def path_of(base: str, variant=None, manifest=None, root: str = "") -> list:
    e = _base(manifest or load_manifest(), base)
    return [resolve(it["file"], root) for it in files_of(e, variant)]


def show_text(base: str, variant=None, manifest=None, root: str = "") -> str:
    e = _base(manifest or load_manifest(), base)
    if e.get("kind") == "source-drop":
        lines = ["source-drop %s: copy into <src>/%s (baseline: %s)" % (base, e.get("target"), e.get("baseline"))]
        for it in e.get("files") or []:
            lines.append("  %s -> %s  sha256=%s" % (it["file"], it.get("dest"), it.get("sha256")))
        return "\n".join(lines) + "\n"
    with open(resolve(variant_entry(e, variant)["file"], root), encoding="utf-8", newline="") as f:
        return f.read()


def _run(cmd, cwd):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as e:
        return 127, str(e)
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


def _apply_git_patch(patch: str, src: str, check: bool, git=None):
    git = shutil.which("git") if git is None else git
    if git:
        rc, _ = _run([git, "apply", "--reverse", "--check", patch], src)
        if rc == 0:
            return RC_OK, "already applied: %s" % os.path.basename(patch)
        rc, out = _run([git, "apply", "--check", patch], src)
        if rc != 0:
            return RC_CHECK, "git apply --check failed:\n%s" % out
        if check:
            return RC_OK, "check ok: %s applies cleanly" % os.path.basename(patch)
        rc, out = _run([git, "apply", patch], src)
        return (RC_OK, "applied %s" % os.path.basename(patch)) if rc == 0 else (RC_CHECK, "git apply failed:\n%s" % out)
    tool = shutil.which("patch")
    if not tool:
        return RC_CHECK, "neither git nor patch found on PATH"
    rc, _ = _run([tool, "-p1", "-R", "--dry-run", "-s", "-f", "-i", patch], src)
    if rc == 0:
        return RC_OK, "already applied: %s" % os.path.basename(patch)
    rc, out = _run([tool, "-p1", "--dry-run", "-s", "-f", "-i", patch], src)
    if rc != 0:
        return RC_CHECK, "patch -p1 --dry-run failed:\n%s" % out
    if check:
        return RC_OK, "check ok: %s applies cleanly" % os.path.basename(patch)
    rc, out = _run([tool, "-p1", "-s", "-f", "-i", patch], src)
    return (RC_OK, "applied %s" % os.path.basename(patch)) if rc == 0 else (RC_CHECK, "patch -p1 failed:\n%s" % out)


def _apply_source_drop(entry: dict, src: str, check: bool, root: str = ""):
    target = os.path.join(src, *str(entry.get("target") or "").split("/"))
    if not os.path.isdir(target):
        return RC_CHECK, "not a %s source tree: %s is missing" % (entry.get("baseline") or "matching", target)
    todo = []
    for it in entry.get("files") or []:
        dest = os.path.join(src, *it["dest"].split("/"))
        if os.path.isfile(dest) and sha256_file(dest) == it["sha256"]:
            continue
        todo.append((resolve(it["file"], root), dest, it["dest"]))
    if not todo:
        return RC_OK, "already applied: every file matches"
    if check:
        return RC_OK, "check ok: %d file(s) would be replaced: %s" % (len(todo), ", ".join(t[2] for t in todo))
    backup = os.path.join(src, ".yee-backup-%s" % time.strftime("%Y%m%d-%H%M%S"))
    n = 0
    while os.path.exists(backup + ("-%d" % n if n else "")):
        n += 1
    backup += "-%d" % n if n else ""
    for srcfile, dest, rel in todo:
        if os.path.isfile(dest):
            b = os.path.join(backup, *rel.split("/"))
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(dest, b)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(srcfile, dest)
    return RC_OK, "replaced %d file(s); originals in %s" % (len(todo), backup)


def apply(base: str, src: str, variant=None, check: bool = False, manifest=None, root: str = "", git=None):
    """Apply (or with check=True only test) one base's patch to the source tree ``src``: (exit code, message).

    ``manifest`` / ``root`` replace the shipped MANIFEST.json and patch directory (tests pass their own);
    ``git`` overrides the git executable ("" = act as if git were missing)."""
    m = manifest or load_manifest()
    e = _base(m, base)
    if not src or not os.path.isdir(src):
        raise PatchUsageError("--src %r is not a directory" % (src,))
    files = files_of(e, variant)
    bad = verify(files, root)
    if bad:
        return RC_CHECK, "refusing to apply, the shipped files do not match MANIFEST.json:\n  " + "\n  ".join(bad)
    if e.get("kind") == "source-drop":
        return _apply_source_drop(e, src, check, root)
    if e.get("kind") == "git-patch":
        return _apply_git_patch(resolve(files[0]["file"], root), os.path.abspath(src), check, git)
    raise PatchUsageError("unknown kind %r" % (e.get("kind"),))


# --------------------------------------------------------------------------------------------- CLI
def cmd_patch(args) -> int:
    try:
        if args.patch_cmd == "list":
            rows = list_rows()
            for r in rows:
                v = ",".join(r["variants"]) if r["variants"] else "-"
                if r["default_variant"]:
                    v += " (default %s)" % r["default_variant"]
                print("%-11s %-12s variants=%s" % (r["base"], r["kind"], v), flush=True)
                print("            baseline: %s" % r["baseline"], flush=True)
                for k, val in sorted(r["env"].items()):
                    print("            env: %s=%s" % (k, val), flush=True)
            return RC_OK
        if args.patch_cmd == "show":
            print(show_text(args.base, args.variant), end="", flush=True)
            return RC_OK
        if args.patch_cmd == "path":
            for p in path_of(args.base, args.variant):
                print(p, flush=True)
            return RC_OK
        if args.patch_cmd == "apply":
            rc, msg = apply(args.base, args.src, args.variant, args.check)
            print(msg, flush=True)
            if rc == RC_OK and not args.check:
                env = _base(load_manifest(), args.base).get("env") or {}
                for k, val in sorted(env.items()):
                    print("then start the server with: export %s=%s" % (k, val), flush=True)
            return rc
    except PatchUsageError as e:
        print("yee patch: %s" % (e,), flush=True)
        return RC_USAGE
    except (OSError, ValueError) as e:
        print("yee patch: %s" % (e,), flush=True)
        return RC_CHECK
    return RC_USAGE


def add_parser(sub) -> None:
    sp = sub.add_parser("patch", help="source patches for TensorFold / cuteafd / mlx-serve")
    ps = sp.add_subparsers(dest="patch_cmd", required=True)
    ps.add_parser("list")
    for name in ("show", "path"):
        p = ps.add_parser(name)
        p.add_argument("base")
        p.add_argument("--variant", default=None)
    p = ps.add_parser("apply")
    p.add_argument("base")
    p.add_argument("--src", required=True, help="root of the source tree to patch")
    p.add_argument("--variant", default=None)
    p.add_argument("--check", action="store_true", help="only check, write nothing")
    sp.set_defaults(fn=cmd_patch)

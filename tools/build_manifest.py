"""Build step: write yee/patches/MANIFEST.json with the sha256 of every shipped patch file.

Run from the package root before ``pip wheel``:  python tools/build_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.path.join(ROOT, "yee", "patches")
SPEC = "src/engines/mlx/spec"


def sha(rel: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(PD, *rel.split("/")), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def item(rel: str, **extra) -> dict:
    d = {"file": rel, "sha256": sha(rel)}
    d.update(extra)
    return d


def main() -> int:
    tf = "tensorfold/0001-recall-after-priced.patch"
    drop_dir = "mlx-serve/files/" + SPEC
    drops = sorted(n for n in os.listdir(os.path.join(PD, *drop_dir.split("/"))) if n.endswith(".zig"))
    manifest = {
        "format": 1,
        "package": "yee 2.2.0",
        "bases": {
            "tensorfold": dict(
                item(tf),
                kind="git-patch",
                baseline="ashhart/TensorFold commit 33ada77c3122fca844a6825316ce8bbabaca05be",
                apply_in="repository root (git apply)",
                env={"TF_RECALL_AFTER_PRICED": "1"},
                rebuild="zig build with the repository's .zig-version (0.17.0 at the baseline)",
                summary="priced proposers (DSv4.1 lookup plan, Llama drafter, Qwen4 device lanes) that offer no "
                        "draft fall back to Recall when TF_RECALL_AFTER_PRICED=1; unset = upstream behaviour",
            ),
            "cuteafd": {
                "kind": "git-patch",
                "baseline": "cuteafd revision 9f53cc6cc0f3ff56de7cb81d587b3fad77137f4c",
                "apply_in": "source root (patch -p1 or git apply)",
                "default_variant": "v2",
                "variants": {"v1": item("cuteafd/0001-recall-v1.patch"),
                             "v2": item("cuteafd/0001-recall-v2.patch")},
                "notes": item("cuteafd/NOTES-s839y-r572.md"),
                "env": {},
                "summary": "DeepSeek V4.1 native serve: Recall copy drafts, greedy only, record in process memory",
            },
            "mlx-serve": {
                "kind": "source-drop",
                "baseline": "M5 xypd mlx-serve source (2026-11-10 r593)",
                "target": SPEC,
                "files": [item(drop_dir + "/" + n, dest=SPEC + "/" + n) for n in drops],
                "env": {},
                "summary": "DSpark Recall for greedy and sampled requests, 8/4-token short-window fallback; "
                           "rebuild mlx-serve after the copy",
            },
        },
    }
    out = os.path.join(PD, "MANIFEST.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s (%d mlx-serve files)" % (out, len(drops)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

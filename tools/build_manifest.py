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
V24_DESTS = (SPEC + "/d_recall.zig", SPEC + "/d_recall_live.zig", SPEC + "/d_recall_cold.zig", "src/env.zig")
VCAP_FILES = ("verify_cap.py", "vcap_policy.py")


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
    kad_dir = "mlx-serve-kad/files/" + SPEC
    kad_drops = sorted(n for n in os.listdir(os.path.join(PD, *kad_dir.split("/"))) if n.endswith(".zig"))
    manifest = {
        "format": 1,
        "package": "yee 4.0.0",
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
            "mlx-serve-2.4": {
                "kind": "source-drop",
                "baseline": "M5 xypd mlx-serve live source with Yee 2.3 (r608 src_base; live binary sha16 "
                            "1ff9c2a5df3f1829)",
                "target": SPEC,
                "files": [item("mlx-serve-2.4/files/" + d, dest=d) for d in V24_DESTS],
                "env": {"MLX_SERVE_D_RECALL_COLD_DIR": "<cold-layer directory on NVMe; unset or empty = 2.3 behaviour>"},
                "built": "mlx-serve.r608-v24, sha16 530781ec80c411bb, version 26.11.1-xypd-v24",
                "rebuild": "zig build check; zig build test -Dtest-filter=d_recall (113/113) and dspark (73/73); "
                           "zig build -Doptimize=ReleaseFast",
                "summary": "Yee 2.4 cold layer: tokens.bin + index.bin on NVMe written by one background thread, "
                           "each round prefetches the next round's 3 branches, the inference thread only does "
                           "non-blocking reads; M5 A/B aggregate decode -0.4852% on average, hit rate "
                           "+4.79/+10.13/+4.36 points, 0 errors",
            },
            "tensorfold-kad": dict(
                item("tensorfold/0002-kad.patch"),
                kind="git-patch",
                baseline="ashhart/TensorFold commit 33ada77c3122fca844a6825316ce8bbabaca05be with "
                         "tensorfold/0001-recall-after-priced.patch applied (apply the tensorfold base first)",
                apply_in="repository root (git apply)",
                env={"TF_KAD": "1"},
                rebuild="zig build with the repository's .zig-version (0.17.0); DeepSeek V4.1 launch width 12: "
                        "\"dspark_block_size\": 12 in the model's config.json",
                summary="KAD: per-stream head draft depth 5..12 within cfg.most_drafts (ladder, hot-reloaded "
                        "policy file, Recall hits widen); TF_KAD=1 or YEE_KAD=1, unset = the 0001 behaviour",
            ),
            "llama.cpp-kad": dict(
                item("llama.cpp-kad/0001-kad.patch"),
                kind="git-patch",
                baseline="ggml-org/llama.cpp v0.9.0 commit cd959bae99ff4bed22bef2604f45241c09bbb8c8",
                apply_in="repository root (git apply or patch -p1)",
                env={"LLAMA_ARG_KAD": "1"},
                rebuild="cmake --build build --target llama-server",
                summary="llama-server KAD: per-slot draft count 5..12 within --spec-draft-n-max (draft-simple, "
                        "EAGLE3, MTP; ngram types native); --kad / LLAMA_ARG_KAD=1 / YEE_KAD=1, --kad-policy; "
                        "unset = upstream",
            ),
            "mlx-serve-kad": {
                "kind": "source-drop",
                "baseline": "mlx-serve-2.4 copied onto the M5 xypd live source (r608 src_base: dspark.zig sha256 "
                            "30412c8bd5a1e1b2..., dspark_group.zig c37b03e31a08b678...); copy mlx-serve-2.4 first",
                "target": SPEC,
                "files": [item(kad_dir + "/" + n, dest=SPEC + "/" + n) for n in kad_drops],
                "env": {"MLX_SERVE_KAD": "1"},
                "rebuild": "zig build check; zig build test -Dtest-filter=d_recall, -Dtest-filter=dspark and "
                           "-Dtest-filter=kad; zig build -Doptimize=ReleaseFast",
                "summary": "DSpark KAD: a stage round verifies at most the request's width 5..12 within "
                           "dspark_block_size (12 in the model's config.json); copy and recall rounds teach the "
                           "ladder; MLX_SERVE_KAD=1 or YEE_KAD=1, unset = 2.4 behaviour",
            },
            "llama.cpp-recall": dict(
                item("llama.cpp-recall/0002-recall.patch"),
                kind="git-patch",
                baseline="ggml-org/llama.cpp v0.9.0 commit cd959bae99ff4bed22bef2604f45241c09bbb8c8 with "
                         "llama.cpp-kad/0001-kad.patch applied (apply llama.cpp-kad first)",
                apply_in="repository root (git apply or patch -p1), after llama.cpp-kad/0001-kad.patch",
                env={"LLAMA_ARG_RECALL": "1"},
                rebuild="cmake --build build --target llama-server",
                summary="llama-server Yee recall: cross-request copy drafts (common/recall.{h,cpp}, C++ port of "
                        "yee.core RecallRecord + RecallEngine, memory or disk record), with or without a draft "
                        "model; a recall proposal makes KAD hot (width 12); --recall / LLAMA_ARG_RECALL=1 / "
                        "RECALL_ACCEL=1, --recall-dir, --recall-read-only, --recall-gib, --recall-n; unset = "
                        "llama.cpp-kad behaviour",
            ),
            "mia-dsv41-vcap": {
                "kind": "source-drop",
                "baseline": "SGLang Mia DeepSeek-V4.1 image (r652/r653 adapter): /opt/dsv41/adapter/verify_cap.py "
                            "sha256 949c3fb72d72b48e... with DSV41_VERIFY_CAP already set",
                "target": ".",
                "files": [item("mia-dsv41-vcap/files/" + n, dest=n) for n in VCAP_FILES],
                "env": {"DSV41_VERIFY_CAP": "conf:0.1", "DSV41_VCAP_CTL": "/state/vcap_ctl.json"},
                "summary": "per-request verify width for the DSpark verify rectangle: while the control "
                           "file does not exist the decision is bit-identical to DSV41_VERIFY_CAP=conf:T; "
                           "mode legacy only observes, mode adapt takes per-request thresholds from "
                           "vcap_policy.VcapPolicy (load / req / pos / recall / temp / explore)",
            },
            "mia-dsv41-image-budget": {
                "kind": "source-drop",
                "baseline": "sglang srt/multimodal/deepseek_v41_image_processing.py sha256 "
                            "dcd06c489aa12fec... (r662 eng/orig)",
                "target": ".",
                "files": [item("mia-dsv41-image-budget/files/srt/multimodal/deepseek_v41_image_processing.py",
                              dest="srt/multimodal/deepseek_v41_image_processing.py",
                              orig_sha256="dcd06c489aa12fecf7a99925d010bb9929b408f834d7e277051bfc3dd41db0ba")],
                "remove_after": ["srt/multimodal/__pycache__/deepseek_v41_image_processing.*.pyc"],
                "env": {"XY_DSV41_IMAGE_BUDGET_RESERVE": "1"},
                "summary": "the image token budget of the decode side reserves the same token the vLLM "
                           "prefill reserves, so a 3024x588 image is 886 tokens on both sides (1024 "
                           "without the patch, the PD request fails with 500); 0 (default) = reference "
                           "behaviour",
            },
            "strata-recall": dict(
                item("strata-recall/0001-yee-recall.patch"),
                kind="git-patch",
                baseline="Niko1221/Strata commit 712fec6b4297c9cfd32ab43daaa2979a9aa01edf (engine 0.1.59); also "
                         "applies on top of strata-0.1.59-expert-server.patch (rtx3080n1 /home/ubuntu/xp/strata-src)",
                apply_in="repository root (git apply or patch -p1)",
                env={"RECALL_ACCEL": "1"},
                rebuild="CUDA build of the strata target (cmake -DSTRATA_ENABLE_CUDA=ON ..., cmake --build <dir> "
                        "--target strata); rtx3080n1: /home/ubuntu/xp/build.sh, then copy build-*/strata to "
                        "/home/ubuntu/xp/engine/strata",
                summary="strata engine Yee recall: cross-request copy drafts (C++ port of yee.core RecallRecord + "
                        "RecallEngine, memory or disk record) replace a window's MTP drafts when the record expects "
                        "more landed tokens; single-stream GEN and batched BGEN slots; --recall / RECALL_ACCEL=1, "
                        "unset = upstream; 3.6.0: replicas share one disk record (every replica appends under "
                        "flock and refreshes every 2 s, STRATA_RECALL_SHARED=0 = the 3.5.x single writer) and a "
                        "two-process unit test tests/yee_recall_shared_test.cpp",
            ),
            "llama.cpp-mtp-vocab": dict(
                item("llama.cpp-mtp-vocab/0001-yee-mtp-vocab.patch"),
                kind="git-patch",
                baseline="ggml-org/llama.cpp commit a3b9c23ead5054832c5ed4b43f652fa1123869a3 (2026-08-21); the two "
                         "anchors in src/models/qwen35.cpp also occur once in v0.9.0 cd959bae",
                apply_in="repository root (git apply or patch -p1)",
                env={"YEE_MTP_VOCAB_N": "131072"},
                rebuild="cmake --build build --target llama-server",
                summary="llama.cpp MTP draft head (qwen35 / qwen35moe nextn graph) computes logits only for token "
                        "ids [0, N) and pads them back to n_vocab with -1000; YEE_MTP_VOCAB_N=N, unset or 0 = "
                        "upstream; drafts are verified by the target, the output does not change",
            ),
        },
    }
    out = os.path.join(PD, "MANIFEST.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s (%d mlx-serve files, %d mlx-serve-2.4 files)" % (out, len(drops), len(V24_DESTS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

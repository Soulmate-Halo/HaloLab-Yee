"""Yee 3.5.2: the llama.cpp-mtp-vocab patch (MTP draft head limited to the first N token ids)."""
import hashlib
import os
import shutil
import subprocess

import pytest

from yee import patches

BASE = "llama.cpp-mtp-vocab"
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "qwen35_a3b9c23.cpp")


def _entry():
    return patches.load_manifest()["bases"][BASE]


def _patch_file():
    import yee
    return os.path.join(os.path.dirname(os.path.abspath(yee.__file__)), "patches", *_entry()["file"].split("/"))


def test_manifest_entry():
    e = _entry()
    assert e["kind"] == "git-patch"
    assert e["file"] == "llama.cpp-mtp-vocab/0001-yee-mtp-vocab.patch"
    assert e["env"] == {"YEE_MTP_VOCAB_N": "131072"}
    assert "a3b9c23ead5054832c5ed4b43f652fa1123869a3" in e["baseline"]
    with open(_patch_file(), "rb") as f:
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest() == e["sha256"]
    assert b"\r\n" not in raw


def test_patch_text():
    with open(_patch_file(), encoding="utf-8") as f:
        text = f.read()
    assert "src/models/qwen35.cpp" in text
    for word in ("YEE_MTP_VOCAB_N", "yee_mtp_vocab_n", "ggml_view_2d", "ggml_pad", "ggml_scale_bias",
                 "[yee-mtp-vocab] MTP draft head limited to first"):
        assert word in text, word
    # one file only: the target graph, the host buffers and the samplers are not touched
    assert text.count("diff --git ") == 1


def _git(args, cwd):
    return subprocess.run(["git", "-c", "core.autocrlf=false"] + args, cwd=cwd, capture_output=True, text=True)


def test_apply_check_revert(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "llama.cpp"
    (repo / "src" / "models").mkdir(parents=True)
    shutil.copyfile(FIXTURE, str(repo / "src" / "models" / "qwen35.cpp"))
    assert _git(["init", "-q"], str(repo)).returncode == 0
    # keep LF in the working tree for every later git call, the ones inside yee.patches included
    assert _git(["config", "core.autocrlf", "false"], str(repo)).returncode == 0
    assert _git(["add", "-A"], str(repo)).returncode == 0
    r = _git(["-c", "user.email=yee@test", "-c", "user.name=yee", "commit", "-q", "-m", "a3b9c23 qwen35.cpp"],
             str(repo))
    assert r.returncode == 0, r.stderr
    rc, msg = patches.apply(BASE, str(repo), check=True)
    assert rc == 0 and "check ok" in msg, msg
    rc, msg = patches.apply(BASE, str(repo))
    assert rc == 0 and "applied 0001-yee-mtp-vocab.patch" in msg, msg
    with open(str(repo / "src" / "models" / "qwen35.cpp"), encoding="utf-8") as f:
        src = f.read()
    assert "yee_mtp_vocab_n(yee_n_vocab)" in src and "YEE_MTP_VOCAB_N" in src
    rc, msg = patches.apply(BASE, str(repo), check=True)
    assert rc == 0 and "already applied" in msg, msg
    r = _git(["apply", "-R", "--check", _patch_file()], str(repo))
    assert r.returncode == 0, r.stderr
    r = _git(["apply", "-R", _patch_file()], str(repo))
    assert r.returncode == 0, r.stderr
    assert _git(["status", "--porcelain"], str(repo)).stdout.strip() == ""

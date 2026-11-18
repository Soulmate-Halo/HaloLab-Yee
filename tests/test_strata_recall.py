"""Yee 3.5.1: the strata engine recall patch (base strata-recall)."""
import os
import re

import pytest

from yee import patches

BASE = "strata-recall"
NEW_FILES = ("include/strata/spec/yee_recall.hpp", "src/spec/yee_recall.cpp",
             "include/strata/spec/yee_recall_strata.hpp", "src/spec/yee_recall_strata.cpp",
             "third_party/nlohmann/include/nlohmann/json.hpp", "tests/yee_recall_test.cpp",
             "tests/yee_recall_strata_test.cpp", "tests/yee_recall_shared_test.cpp")  # the last one: 3.6.0
CHANGED = ("cmake/backends/cpu.cmake", "src/program/generate.cpp")


def _entry():
    return patches.load_manifest()["bases"][BASE]


def _text():
    with open(patches.path_of(BASE)[0], encoding="utf-8", newline="") as f:
        return f.read()


def test_manifest_entry():
    e = _entry()
    assert e["kind"] == "git-patch"
    assert e["file"] == "strata-recall/0001-yee-recall.patch"
    assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"])
    assert "712fec6b4297c9cfd32ab43daaa2979a9aa01edf" in e["baseline"]
    assert e["env"] == {"RECALL_ACCEL": "1"}
    for k in ("apply_in", "rebuild", "summary"):
        assert e.get(k), k


def test_patch_matches_its_sha256():
    e = _entry()
    assert patches.verify([{"file": e["file"], "sha256": e["sha256"]}]) == []


def test_list_and_show():
    rows = {r["base"]: r for r in patches.list_rows()}
    assert rows[BASE]["kind"] == "git-patch"
    assert rows[BASE]["env"] == {"RECALL_ACCEL": "1"}
    assert patches.show_text(BASE).startswith("diff --git ")


def test_patch_is_the_contract_diff():
    text = _text()
    if "yee_recall_placeholder.hpp" in text:
        pytest.skip("placeholder patch: the build ran without the real strata-recall patch")
    assert "\r\n" not in text
    files = re.findall(r"^diff --git a/(\S+) b/\1$", text, re.M)
    for name in NEW_FILES:
        assert name in files, name
        assert re.search(r"^diff --git a/%s b/%s\nnew file mode " % (re.escape(name), re.escape(name)), text, re.M), name
    for name in CHANGED:
        assert name in files, name
    assert len(files) == len(NEW_FILES) + len(CHANGED), files
    # the engine flags and the switch the docs name
    for word in ('"--recall"', '"--no-recall"', '"--recall-dir"', '"--recall-gib"', '"--recall-read-only"',
                 "RECALL_ACCEL=1", "[yee-recall] strata: on", "STRATA_RECALL_SHARED"):
        assert word in text, word


def test_check_refuses_a_tree_that_is_not_strata(tmp_path):
    rc, msg = patches.apply(BASE, str(tmp_path), check=True)
    assert rc == patches.RC_CHECK, msg
    assert os.listdir(str(tmp_path)) == []

"""KAD on transformers assisted generation, with stand-in candidate generators (no torch needed).

The stand-ins behave like transformers 5.22.2: the assistant drafts ``num_assistant_tokens`` (prompt lookup:
``num_output_tokens``) per round and ``update_candidate_strategy`` gets the target logits of v + 1 positions.
The real-model check is in the r641 verify script.
"""
import gc
import os
import types

import pytest

from yee.integrations import hf_kad as hk


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAssisted:
    def __init__(self, n=12, schedule="constant"):
        self.num_assistant_tokens = n
        self.assistant_model = types.SimpleNamespace(
            generation_config=Cfg(num_assistant_tokens=n, num_assistant_tokens_schedule=schedule))
        self.assistant_generation_config = Cfg(num_assistant_tokens=n, num_assistant_tokens_schedule=schedule)
        self.seen = []

    def get_candidates(self, input_ids, **kwargs):
        self.seen.append(self.num_assistant_tokens)
        return input_ids, None

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        if self.assistant_generation_config.num_assistant_tokens_schedule in ("heuristic", "heuristic_transient"):
            if num_matches == len(scores[0]) - 1:
                self.num_assistant_tokens += 2
            else:
                self.num_assistant_tokens = max(1, self.num_assistant_tokens - 1)


class FakeLookup:
    def __init__(self, n=12):
        self.num_output_tokens = n
        self.seen = []

    def get_candidates(self, input_ids, **kwargs):
        self.seen.append(self.num_output_tokens)
        return input_ids, None

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        pass


class MTPCandidateGenerator:  # same class name as the transformers native generator
    def get_candidates(self, input_ids, **kwargs):
        return input_ids, None

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        pass


def scores(v):
    return [[0.0] * (v + 1)]


def rounds(cg, accepts):
    """Drive cg like _assisted_decoding: get_candidates, then update with v = drafted, a = accepts(v)."""
    for acc in accepts:
        cg.get_candidates([1, 2, 3])
        v = cg.seen[-1]
        cg.update_candidate_strategy(None, scores(v), acc(v) if callable(acc) else acc)


ALL = (lambda v: v)
YEE = Cfg(num_assistant_tokens_schedule=hk.SCHEDULE)
PLAIN = Cfg(num_assistant_tokens_schedule="constant")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("YEE_KAD")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path / "kad"))
    monkeypatch.setenv("YEE_KAD_LOG_S", "3600")
    monkeypatch.setattr(hk, "_KAD", None)
    monkeypatch.setattr(hk, "_WARNED", set())
    monkeypatch.setattr(hk, "_NOTED", set())
    return tmp_path


def test_off_without_opt_in():
    cg = FakeAssisted(12)
    assert hk.attach(cg, PLAIN) == "off"
    assert "get_candidates" not in cg.__dict__ and "update_candidate_strategy" not in cg.__dict__


def test_assistant_schedule_opt_in_climbs_to_12():
    cg = FakeAssisted(12, hk.SCHEDULE)
    assert hk.attach(cg, PLAIN) == "kad"
    rounds(cg, [ALL] * 6)
    assert cg.seen == [5, 7, 9, 11, 12, 12]
    assert cg.num_assistant_tokens == 12  # launch count restored between rounds


def test_generate_kwarg_opt_in():
    cg = FakeAssisted(12)
    assert hk.attach(cg, YEE) == "kad"
    rounds(cg, [ALL, ALL, 3, 0])
    assert cg.seen == [5, 7, 9, 5]  # 9 accepted 3 -> max(5, 3 + 1)


def test_env_on_replaces_heuristic_without_touching_user_config(monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    cg = FakeAssisted(12, "heuristic")
    assert hk.attach(cg, PLAIN) == "kad"
    assert cg.assistant_generation_config.num_assistant_tokens_schedule == "constant"
    assert cg.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
    rounds(cg, [ALL] * 5)
    assert cg.seen == [5, 7, 9, 11, 12]
    assert cg.num_assistant_tokens == 12  # what transformers persists for "heuristic"


def test_env_off_wins_over_schedule(monkeypatch):
    monkeypatch.setenv("YEE_KAD", "0")
    cg = FakeAssisted(12, hk.SCHEDULE)
    assert hk.attach(cg, YEE) == "off"


def test_rejections_stay_at_5():
    cg = FakeAssisted(12, hk.SCHEDULE)
    hk.attach(cg, PLAIN)
    rounds(cg, [0] * 5)
    assert cg.seen == [5] * 5


def test_launch_20_is_capped_at_12_and_warns(capsys):
    cg = FakeAssisted(20, hk.SCHEDULE)
    hk.attach(cg, PLAIN)
    rounds(cg, [ALL] * 7)
    assert cg.seen == [5, 7, 9, 11, 12, 12, 12]
    cg2 = FakeAssisted(20, hk.SCHEDULE)
    hk.attach(cg2, PLAIN)
    assert capsys.readouterr().out.count("WARNING launch K=20, KAD needs 12") == 1


def test_prompt_lookup_and_rounds_without_candidates():
    cg = FakeLookup(12)
    assert hk.attach(cg, YEE) == "kad"
    cg.get_candidates([1])
    cg.update_candidate_strategy(None, scores(0), 0)  # no n-gram match: nothing verified, state unchanged
    rounds(cg, [ALL, ALL])
    assert cg.seen == [5, 5, 7]
    assert cg.num_output_tokens == 12


def test_native_generator_is_statistics_only():
    cg = MTPCandidateGenerator()
    assert hk.attach(cg, YEE) == "native"
    assert "get_candidates" not in cg.__dict__
    cg.update_candidate_strategy(None, scores(3), 3)
    snap = hk.kad().snapshot()
    assert snap["v_steps"] == 1 and snap["sum_a"] == 3


def test_state_released_when_the_generator_is_gone():
    cg = FakeAssisted(12, hk.SCHEDULE)
    hk.attach(cg, PLAIN)
    rounds(cg, [ALL])
    assert hk.kad().snapshot()["tracked"] == 1
    del cg
    gc.collect()
    assert hk.kad().snapshot()["tracked"] == 0


def test_width_error_keeps_launch_count(monkeypatch):
    cg = FakeAssisted(12, hk.SCHEDULE)
    hk.attach(cg, PLAIN)

    def boom(rid, structured=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(hk.kad(), "width", boom)
    rounds(cg, [ALL])
    assert cg.seen == [12] and hk.kad().snapshot()["errors"] >= 1


def test_attach_twice_is_a_no_op():
    cg = FakeAssisted(12, hk.SCHEDULE)
    assert hk.attach(cg, PLAIN) == "kad"
    assert hk.attach(cg, PLAIN) == "off"


def test_install_on_real_transformers():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    ok, why = hk.install()
    try:
        assert ok, why
        assert hk.installed()
        assert all(row[1] for row in hk.check())
    finally:
        hk.uninstall()
    assert not hk.installed()

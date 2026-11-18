"""KAD on mlx-lm: the source edit and its runtime, on a pure-Python stand-in of speculative_generate_step.

The stand-in keeps every line the edit anchors on (same control flow as mlx-lm 0.32.2), so these tests need
neither mlx nor mlx-lm. The real-model check is in the r641 verify script.
"""
import importlib.util
import inspect
import json
import os
import sys
import textwrap

import pytest

from yee.integrations import mlx_lm_kad as mk

FAKE = '''
import contextlib

CALLS = []


class mx:
    @staticmethod
    def stream(s):
        return contextlib.nullcontext()


def _rewind_cache(num_draft, num_accept):
    pass


def speculative_generate_step(prompt, model, draft_model, stream=None, *, num_draft_tokens=2, max_tokens=256):
    """Stand-in with the anchor lines of mlx_lm.generate.speculative_generate_step."""
    y = list(prompt)
    with mx.stream(stream):
        ntoks = 0
        num_draft = 0
        n = 0
        try:
            while True:
                num_draft = min(max_tokens - ntoks, num_draft_tokens)
                ctx = list(y)
                draft_tokens = []
                for _ in range(num_draft):
                    t = draft_model(ctx)
                    draft_tokens.append(t)
                    ctx.append(t)
                ctx = list(y)
                tokens = []
                for i in range(num_draft + 1):
                    tokens.append(model(ctx))
                    if i < num_draft:
                        ctx.append(draft_tokens[i])
                CALLS.append(num_draft)
                n = 0
                while n < num_draft:
                    if tokens[n] != draft_tokens[n]:
                        break
                    y.append(tokens[n])
                    n += 1
                    ntoks += 1
                    yield y[-1], None, True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    y.append(tokens[n])
                    yield tokens[n], None, False
                if ntoks == max_tokens:
                    break
        finally:
            _rewind_cache(num_draft, n)
'''

PROMPT = (1, 2, 3)


def target(ctx):
    return (sum(ctx) * 7 + len(ctx) * 3) % 11


def wrong(ctx):
    return (target(ctx) + 1) % 11


def greedy(n, prompt=PROMPT):
    ctx, out = list(prompt), []
    for _ in range(n):
        out.append(target(ctx))
        ctx.append(out[-1])
    return out


def _load(path, name, src):
    path.write_text(textwrap.dedent(src), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("YEE_KAD")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path / "kad"))
    monkeypatch.setenv("YEE_KAD_LOG_S", "3600")
    monkeypatch.setattr(mk, "_KAD", None)
    monkeypatch.setattr(mk, "_ORIG", {})
    monkeypatch.setattr(mk, "_WARNED", set())
    return tmp_path


@pytest.fixture
def fake(env):
    mod = _load(env / "fake_mlx_generate.py", "fake_mlx_generate", FAKE)
    yield mod
    sys.modules.pop("fake_mlx_generate", None)


def run(mod, draft, k=12, max_tokens=40):
    del mod.CALLS[:]
    gen = mod.speculative_generate_step(list(PROMPT), target, draft, num_draft_tokens=k, max_tokens=max_tokens)
    return [t for t, _, _ in gen], list(mod.CALLS)


def test_rewrite_has_the_five_edits(fake):
    new = mk.rewrite(inspect.getsource(fake.speculative_generate_step))
    assert "def _yee_kad_speculative_generate_step(" in new
    assert "_yk_rid = _yee_kad_begin(num_draft_tokens)" in new
    assert "num_draft_tokens, _yee_kad_width(_yk_rid, num_draft_tokens))" in new
    assert "_yee_kad_observe(_yk_rid, num_draft, n)" in new
    assert "_yee_kad_end(_yk_rid)" in new
    compile(new, "<t>", "exec")


def test_install_off_without_env(fake):
    assert mk.install() == (False, "YEE_KAD is off")
    assert not getattr(fake.speculative_generate_step, mk.MARK, False)


def test_full_acceptance_climbs_to_12_and_output_is_unchanged(fake):
    base_out, base_calls = run(fake, target, max_tokens=60)
    assert base_calls == [12, 12, 12, 12, 8]
    ok, why = mk.install_into(fake)
    assert ok, why
    out, calls = run(fake, target, max_tokens=60)
    assert out == base_out == greedy(60)
    assert calls == [5, 7, 9, 11, 12, 11]  # the last round is cut by max_tokens
    snap = mk.kad().snapshot()
    assert snap["tracked"] == 0 and snap["errors"] == 0


def test_rejections_keep_the_width_at_5(fake):
    mk.install_into(fake)
    out, calls = run(fake, wrong)
    assert out == greedy(40)
    assert set(calls[:-4]) == {5}


def test_climbs_then_drops_after_a_rejection(fake):
    def draft(ctx):
        return target(ctx) if len(ctx) < len(PROMPT) + 45 else wrong(ctx)

    mk.install_into(fake)
    out, calls = run(fake, draft, max_tokens=80)
    assert out == greedy(80)
    # 5, 7, 9, 11 all accepted; 12 accepts 9 -> 10; 10 accepts 0 -> 5; then 5 until max_tokens cuts the tail
    assert calls[:7] == [5, 7, 9, 11, 12, 10, 5]
    assert set(calls[6:-4]) == {5} and calls[-4:] == [4, 3, 2, 1]


def test_launch_k_caps_the_width_and_warns_once(fake, capsys):
    mk.install_into(fake)
    run(fake, target, k=3)
    out, calls = run(fake, target, k=3)
    assert out == greedy(40) and max(calls) == 3
    assert capsys.readouterr().out.count("WARNING launch K=3, KAD needs 12") == 1


def test_policy_file_fixed_7(fake, env, monkeypatch):
    pol = env / "kad_policy.json"
    pol.write_text(json.dumps({"mode": "fixed", "fixed": 7}), encoding="utf-8")
    monkeypatch.setenv("YEE_KAD_POLICY", str(pol))
    mk.install_into(fake)
    out, calls = run(fake, target)
    assert out == greedy(40) and set(calls[:-1]) == {7}


def test_width_error_falls_back_to_launch_k(fake, monkeypatch):
    mk.install_into(fake)

    def boom(rid, structured=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(mk.kad(), "width", boom)
    out, calls = run(fake, target)
    assert out == greedy(40) and set(calls[:-1]) == {12}
    assert mk.kad().snapshot()["errors"] >= 1


def test_missing_anchor_installs_nothing(env):
    src = FAKE.replace("                if ntoks < max_tokens:\n", "                if ntoks <= max_tokens - 1:\n")
    mod = _load(env / "fake_mlx_generate2.py", "fake_mlx_generate2", src)
    try:
        before = mod.speculative_generate_step
        ok, why = mk.install_into(mod)
        assert not ok and "anchors" in why
        assert mod.speculative_generate_step is before
    finally:
        sys.modules.pop("fake_mlx_generate2", None)


def test_idempotent_and_uninstall(fake):
    orig = fake.speculative_generate_step
    assert mk.install_into(fake)[0]
    assert mk.install_into(fake) == (True, "already installed")
    assert fake.speculative_generate_step.__wrapped__ is orig
    assert mk.uninstall()
    assert fake.speculative_generate_step is orig

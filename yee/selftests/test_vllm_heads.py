"""Plain-assert tests: vllm_spec wraps every V2 speculator's propose once (EAGLE / MTP / DFlash2 / Gemma4 ...).

Run: python tests/test_vllm_heads.py. The full decode simulation of the hook is in test_vllm_hook.py (DFlash
signature); here fake speculator classes check which classes get wrapped, that a subclass calling
super().propose runs the hook once, and the module table. With the real vLLM importable it also patches the real
speculator modules that exist in that build.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "yee")):
    sys.path.insert(0, ROOT)

from yee.integrations import vllm_spec as vs  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def _fake_classes():
    class EagleSpeculator:
        def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                    num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None,
                    dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
            return "eagle"

    class MTPSpeculator(EagleSpeculator):
        def propose(self, *a, **kw):
            return "mtp+" + super().propose(*a, **kw)

    class Gemma4Speculator(EagleSpeculator):
        pass

    return EagleSpeculator, MTPSpeculator, Gemma4Speculator


@test
def test_wraps_own_propose_only():
    E, M, G4 = _fake_classes()
    assert vs._patch_speculator_class(E, "m")
    assert vs._patch_speculator_class(M, "m")
    assert not vs._patch_speculator_class(G4, "m"), "inherits a wrapped propose: nothing to do"
    assert not vs._patch_speculator_class(E, "m"), "idempotent"
    assert E.propose.__recall_wrapped__ and M.propose.__recall_wrapped__


@test
def test_hook_runs_once_through_super():
    E, M, _ = _fake_classes()
    vs._patch_speculator_class(E, "m")
    vs._patch_speculator_class(M, "m")
    seen = []
    orig = vs.participates

    def spy(runner, ib, out, dummy, prof):
        seen.append(out)
        return False

    vs.participates = spy
    try:
        args = (None,) * 11
        assert M().propose(*args) == "mtp+eagle"
        assert seen == ["mtp+eagle"], seen
        seen.clear()
        assert E().propose(*args) == "eagle"
        assert seen == ["eagle"], seen
    finally:
        vs.participates = orig
    assert vs._state.get("in_propose") is False


@test
def test_module_table():
    for name in ("eagle", "mtp", "multi_module_mtp", "gemma4", "dflash", "dflash2", "dspark"):
        assert "vllm.v1.worker.gpu.spec_decode.%s.speculator" % name in vs.TARGETS, name
    assert "vllm.v1.worker.gpu.model_runner" in vs.TARGETS
    mod = types.ModuleType("vllm.v1.worker.gpu.spec_decode.eagle.speculator")
    E, _, _ = _fake_classes()
    mod.EagleSpeculator = E
    vs._patch_speculator_module(mod)
    assert E.propose.__recall_wrapped__


@test
def test_real_vllm_speculators():
    try:
        import vllm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return "SKIP real vLLM not importable here (%s)" % type(e).__name__
    patched = []
    for mname, classes in vs.SPECULATORS.items():
        try:
            m = importlib.import_module(mname)
        except Exception:  # noqa: BLE001
            continue
        vs._patch_speculator_module(m)
        for c in classes:
            cls = getattr(m, c, None)
            if cls is not None:
                assert getattr(cls.propose, "__recall_wrapped__", False), (mname, c)
                patched.append(c)
    assert "DFlashSpeculator" in patched or "DSparkSpeculator" in patched, patched
    print("    real speculators wrapped: %s" % ", ".join(patched))


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

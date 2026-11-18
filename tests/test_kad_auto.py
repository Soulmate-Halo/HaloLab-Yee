"""YEE_KAD=1 autoload: adapters install when their trigger module is imported, never before, never raising."""
import os
import sys
import types

import pytest

from yee import autoload
from yee.integrations import kad_auto


@pytest.fixture
def auto(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("YEE_KAD"):
            monkeypatch.delenv(name, raising=False)
    calls = []

    def adapter(name, ok=True, boom=False):
        mod = types.ModuleType("yee.integrations." + name)

        def install(enabled=None):
            calls.append((name, enabled))
            if boom:
                raise RuntimeError("boom")
            return ok, "reason"

        mod.install = install
        monkeypatch.setitem(sys.modules, "yee.integrations." + name, mod)

    engine = tmp_path / "fake_engine"
    engine.mkdir()
    (engine / "__init__.py").write_text("", encoding="utf-8")
    (engine / "worker.py").write_text("LOADED = True\n", encoding="utf-8")
    (engine / "other.py").write_text("LOADED = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("fake_engine", "fake_engine.worker", "fake_engine.other"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    kad_auto.uninstall()
    yield types.SimpleNamespace(calls=calls, adapter=adapter)
    kad_auto.uninstall()
    for name in ("fake_engine", "fake_engine.worker", "fake_engine.other"):
        sys.modules.pop(name, None)


def test_off_arms_nothing(auto, monkeypatch):
    monkeypatch.setattr(kad_auto, "TRIGGERS", (("fake_engine.worker", "fake_a"),))
    auto.adapter("fake_a")
    before = list(sys.meta_path)
    assert kad_auto.install() == []
    assert sys.meta_path == before
    assert kad_auto.status()["armed"] is False
    assert autoload.kad_enabled() is False


def test_install_waits_for_the_trigger_module(auto, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "1")
    monkeypatch.setattr(kad_auto, "TRIGGERS", (("fake_engine.worker", "fake_a"), ("fake_engine.other", "fake_b")))
    auto.adapter("fake_a")
    auto.adapter("fake_b", ok=False)
    assert kad_auto.install() == ["fake_a", "fake_b"]
    assert auto.calls == []
    assert kad_auto.status()["waiting"] == ["fake_a", "fake_b"]
    import fake_engine.worker  # noqa: F401
    assert auto.calls == [("fake_a", True)]
    assert sys.modules["fake_engine.worker"].LOADED is True
    import fake_engine.other  # noqa: F401
    assert auto.calls == [("fake_a", True), ("fake_b", True)]
    st = kad_auto.status()
    assert st["waiting"] == [] and st["installed"] == ["fake_a"] and st["failed"] == ["fake_b"]
    assert kad_auto.install() == []  # armed once per process


def test_already_imported_installs_at_once_and_errors_are_contained(auto, monkeypatch, capsys):
    monkeypatch.setenv("YEE_KAD", "1")
    import fake_engine.worker  # noqa: F401
    monkeypatch.setattr(kad_auto, "TRIGGERS", (("fake_engine.worker", "fake_boom"),))
    auto.adapter("fake_boom", boom=True)
    assert kad_auto.install() == ["fake_boom"]
    assert auto.calls == [("fake_boom", True)]
    assert "[yee-kad] autoload of fake_boom failed" in capsys.readouterr().out
    assert kad_auto.status()["failed"] == ["fake_boom"]


def test_autoload_main_arms_kad(auto, monkeypatch):
    monkeypatch.setenv("YEE_KAD", "on")
    monkeypatch.delenv("RECALL_ACCEL", raising=False)
    monkeypatch.setattr(kad_auto, "TRIGGERS", (("fake_engine.worker", "fake_a"),))
    auto.adapter("fake_a")
    assert autoload.kad_enabled() is True
    assert autoload.main() == 0
    assert kad_auto.status()["waiting"] == ["fake_a"]


def test_triggers_name_real_adapters():
    import importlib

    for trigger, adapter in kad_auto.TRIGGERS:
        mod = importlib.import_module("yee.integrations." + adapter)
        assert callable(getattr(mod, "install", None)), adapter
        assert trigger.split(".")[0] in ("sglang", "tensorrt_llm", "lmdeploy", "mlx_lm", "transformers")

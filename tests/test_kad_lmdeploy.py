"""LMDeploy is native for KAD: install() only reports it and never patches anything."""
import os

from yee.integrations import lmdeploy_kad as lk


def _clean(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("YEE_KAD"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YEE_KAD_DIR", str(tmp_path))


def test_native_off_is_silent(monkeypatch, tmp_path, capsys):
    _clean(monkeypatch, tmp_path)
    ok, why = lk.install()
    assert ok is False and "native" in why
    assert capsys.readouterr().out == ""


def test_native_on_prints_reason(monkeypatch, tmp_path, capsys):
    _clean(monkeypatch, tmp_path)
    monkeypatch.setenv("YEE_KAD", "1")
    ok, why = lk.install()
    out = capsys.readouterr().out
    assert ok is False and "native" in why
    assert out.startswith("[yee-kad] lmdeploy native: ")
    assert "12 speculative tokens" in out


def test_status_reports_native():
    st = lk.status()
    assert st["mode"] == "native" and st["installed"] is False
    assert st["base"] == "lmdeploy" and "0.18.0" in st["verified"]

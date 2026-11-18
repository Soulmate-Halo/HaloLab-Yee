"""YEE_KAD=1 autoload: install each Python engine's KAD adapter when that engine's module is imported.

``import yee.autoload`` (the .pth written by ``yee install-hook``) calls :func:`install` at interpreter start when
YEE_KAD is 1 / true / yes / on. Nothing is imported here: a finder on ``sys.meta_path`` waits for the trigger module
of each adapter and calls the adapter's ``install()`` right after that module ran (the adapter imports the other
modules it needs then; it checks versions and symbols itself and prints one line when it does not install).

Adapters: SGLang (``sglang_kad``), TensorRT-LLM (``trtllm_kad``), LMDeploy (``lmdeploy_kad``, native: prints one
line), mlx-lm (``mlx_lm_kad``), transformers (``hf_kad``). vLLM keeps its own switch ``VLLM_D_RECALL_KAD``
(``vllm_d_recall``); llama.cpp, TensorFold and mlx-serve are patches (``yee patch``). Never raises.
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

from .. import kad as _core

# (module whose import triggers the install, adapter module in yee.integrations)
TRIGGERS = (
    ("sglang.srt.speculative.eagle_worker_v2", "sglang_kad"),
    ("tensorrt_llm._torch.pyexecutor.py_executor", "trtllm_kad"),
    ("lmdeploy.messages", "lmdeploy_kad"),
    ("mlx_lm.generate", "mlx_lm_kad"),
    ("transformers.generation.utils", "hf_kad"),
)

_STATE = {"finder": None, "installed": [], "failed": []}


def _hook(adapter: str):
    def run(_module):
        try:
            mod = importlib.import_module("%s.%s" % (__package__, adapter))
            out = mod.install(True)
            ok = bool(out[0]) if isinstance(out, tuple) else bool(out)
            (_STATE["installed"] if ok else _STATE["failed"]).append(adapter)
        except Exception as e:  # noqa: BLE001  never take the engine down
            _STATE["failed"].append(adapter)
            print("%s autoload of %s failed (pid %d): %r" % (_core.TAG, adapter, os.getpid(), e), flush=True)

    run.adapter = adapter
    return run


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, hook):
        self.inner = inner
        self.hook = hook

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.hook(module)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of each trigger module once so that its hook runs right after the module executed."""

    def __init__(self, hooks: dict):
        self.hooks = dict(hooks)

    def find_spec(self, fullname, path=None, target=None):
        hook = self.hooks.get(fullname)
        if hook is None:
            return None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                self.hooks.pop(fullname, None)
                spec.loader = _Loader(spec.loader, hook)
            return spec
        return None


def install(enabled=None) -> list:
    """Arm the adapters (an already imported trigger module installs at once). Returns the adapter names armed."""
    on = _core.env_enabled() if enabled is None else bool(enabled)
    if not on or _STATE["finder"] is not None:
        return []
    hooks = {}
    armed = []
    for trigger, adapter in TRIGGERS:
        hook = _hook(adapter)
        armed.append(adapter)
        if trigger in sys.modules:
            hook(sys.modules[trigger])
        else:
            hooks[trigger] = hook
    finder = _Finder(hooks)
    _STATE["finder"] = finder
    if hooks:
        sys.meta_path.insert(0, finder)
    return armed


def uninstall() -> None:
    """Remove the finder (tests); adapters already installed stay installed."""
    finder = _STATE["finder"]
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
    _STATE["finder"] = None
    _STATE["installed"] = []
    _STATE["failed"] = []


def status() -> dict:
    finder = _STATE["finder"]
    waiting = sorted(h.adapter for h in finder.hooks.values()) if finder is not None else []
    return {"armed": finder is not None, "waiting": waiting, "installed": list(_STATE["installed"]),
            "failed": list(_STATE["failed"])}


__all__ = ["TRIGGERS", "install", "uninstall", "status"]

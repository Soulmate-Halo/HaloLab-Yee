"""Compatibility name of Yee: recall-accel 2.1 was renamed ``yee`` in 2.2.0.

``import recall_accel`` returns the ``yee`` package itself, and every ``recall_accel.<sub>`` import resolves to
the very same module object as ``yee.<sub>`` (one module, two names), so deployments that hard-coded
``import recall_accel.autoload`` or ``recall_accel.integrations.sglang_dspark`` keep working unchanged.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_OLD = "recall_accel"
_NEW = "yee"


class _AliasLoader(importlib.abc.Loader):
    """Hands back the already-imported ``yee.<sub>`` module under the old name."""

    def __init__(self, target: str):
        self.target = target
        self._spec = None

    def create_module(self, spec):
        mod = importlib.import_module(self.target)
        self._spec = getattr(mod, "__spec__", None)
        return mod

    def exec_module(self, module):
        # module_from_spec() stamped the alias spec on the real module: put the real one back
        if self._spec is not None:
            module.__spec__ = self._spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new = _NEW + fullname[len(_OLD):]
        try:
            real = importlib.util.find_spec(new)
        except (ImportError, AttributeError, ValueError):
            return None
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new),
                                               is_package=real.submodule_search_locations is not None)


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

for _name, _mod in list(sys.modules.items()):  # yee submodules imported before this package
    if _name == _NEW or _name.startswith(_NEW + "."):
        sys.modules.setdefault(_OLD + _name[len(_NEW):], _mod)

sys.modules[_OLD] = importlib.import_module(_NEW)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Have every rank's executor worker print what its engine runs, as it sets the engine up.

The gates put this directory first on ``PYTHONPATH`` and set
``MV2FLOW_RANK_FACTS=1`` (``check_runtime.RANK_FACTS_ENV``), so this
``sitecustomize`` runs at interpreter start in every process of the gate's
job. In each it waits for ``tensorrt_llm.executor.base_worker`` to be
imported and wraps ``BaseWorker.setup_engine``: once the worker's engine is
set up, the worker prints one ``MV2FLOW_RANK_ENGINE`` record carrying its
rank (``engine_facts.emit_rank``).

Without the variable this module only hands on to the next ``sitecustomize``
on ``sys.path``, the one it shadows; that hand-on moves forward along
``sys.path`` only, so two hooks of this kind never hand on to each other in a
loop.

Standard library only.
"""

import importlib.util
import os
import sys

_HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
_ENABLED = os.environ.get("MV2FLOW_RANK_FACTS") == "1"
_MODULE = "tensorrt_llm.executor.base_worker"
_FACTS = os.path.join(os.path.dirname(_HERE), "engine_facts.py")


def _report(worker):
    spec = importlib.util.spec_from_file_location("mv2flow_engine_facts", _FACTS)
    facts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facts)
    facts.emit_rank(worker)


def _wrap(module):
    cls = getattr(module, "BaseWorker", None)
    original = getattr(cls, "setup_engine", None)
    if original is None or getattr(original, "_mv2flow_observer", False):
        return

    def setup_engine(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        try:
            _report(worker)
        # An observation of an engine that is up must never stop it.
        except Exception as exc:  # noqa: BLE001
            try:
                sys.stderr.write(f"[mv2flow-rank-facts] not read: {type(exc).__name__}: {exc}\n")
            except (OSError, ValueError):
                pass
        return result

    setup_engine.__wrapped__ = original
    setup_engine._mv2flow_observer = True
    cls.setup_engine = setup_engine


class _Loader:
    """The module's own loader, with :func:`_wrap` run once the module is executed."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _wrap(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Watcher:
    """Find :data:`_MODULE` with the other finders and hand back its spec, loader wrapped."""

    def find_spec(self, name, path=None, target=None):
        if name != _MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _Loader(spec.loader)
            return spec
        return None


def _chain():
    """Run the next ``sitecustomize`` on ``sys.path`` after this one's directory."""
    entries = [os.path.realpath(os.path.abspath(e or ".")) for e in sys.path]
    start = entries.index(_HERE) + 1 if _HERE in entries else 0
    for directory in entries[start:]:
        if directory == _HERE:
            continue
        candidate = os.path.join(directory, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


if _ENABLED:
    sys.meta_path.insert(0, _Watcher())
_chain()

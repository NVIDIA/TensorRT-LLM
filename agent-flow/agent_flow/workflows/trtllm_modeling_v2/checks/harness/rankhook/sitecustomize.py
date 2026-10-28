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

With ``MV2FLOW_READS_DIR`` and ``MV2FLOW_TREE`` set, every process also
writes which of the tree's Python modules it imported (:func:`_dump_reads`),
once the engine is up and at exit.

Without the variable this module only hands on to the next ``sitecustomize``
on ``sys.path``, the one it shadows; that hand-on moves forward along
``sys.path`` only, so two hooks of this kind never hand on to each other in a
loop.

Standard library only.
"""

import atexit
import importlib.util
import json
import os
import socket
import sys

_HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
_ENABLED = os.environ.get("MV2FLOW_RANK_FACTS") == "1"
_MODULE = "tensorrt_llm.executor.base_worker"
_FACTS = os.path.join(os.path.dirname(_HERE), "engine_facts.py")
#: Where each process writes what it read of the tree under test (``MV2FLOW_TREE``).
_READS_DIR = os.environ.get("MV2FLOW_READS_DIR", "")
_TREE = os.environ.get("MV2FLOW_TREE", "")


def _dump_reads():
    """Write the tree's modules this process imported, relative to the tree, to ``_READS_DIR``.

    One file per process, rewritten whole each time with everything imported
    so far: once the engine is up, and again at exit (``run/workspace/snapshots.py``
    reads them).
    """
    if not (_READS_DIR and _TREE):
        return
    root = os.path.realpath(_TREE) + os.sep
    found = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        real = os.path.realpath(path)
        if real.startswith(root) and real.endswith(".py"):
            found.add(real[len(root) :])
    target = os.path.join(_READS_DIR, f"{socket.gethostname()}-{os.getpid()}.json")
    try:
        os.makedirs(_READS_DIR, exist_ok=True)
        partial = f"{target}.{os.getpid()}.tmp"
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(sorted(found), handle)
        os.replace(partial, target)
    # What a process read is an observation; failing to write it never stops the process.
    except (OSError, ValueError):
        pass


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
        _dump_reads()
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
if _READS_DIR:
    atexit.register(_dump_reads)
_chain()
